"""Free-pool cap on large-image heavy memory stages.

This is not a global request limiter and it does not change per-account
image concurrency. It only bounds work that allocates large image buffers
(download, decode, Lanczos upscale, base64 expansion) inside this process.

The guard is off unless this deployment is the free pool. Plus and Adobe
processes that do not mount ``services.free_pool_policy`` and do not set
``CHATGPT2API_FREE_LARGE_IMAGE_GUARD=1`` never take a permit. Setting the
variable to ``0`` forces the guard off even on a free deployment.

Scope is one OS process. ``Dockerfile`` starts a single uvicorn worker
(``uvicorn main:app`` with no ``--workers``). Threads and event loops in
that process share one lock. A second worker or container has its own cap
of ``max_active`` and is not covered. This does not bound memory in the
shared image-size proxy, Plus, or Adobe.

A permit is acquired on the worker thread that performs the allocation,
before the large bytes are read or decoded. Cancelling the awaiting
coroutine does not release that permit while the thread is still inside
the heavy section. Queue admission happens before upstream generation:
a full queue or an admission timeout raises ``LargeImageAdmissionError``
and must not start a generation. A timeout while entering the heavy
section after generation has already started must fail that same attempt
and must not launch another generation.

Defaults (free pool only):

- active heavy stages: 5
- jobs waiting to run or already admitted but not in a heavy stage: 10
- admission / heavy-entry wait: 180s

180s is one image download timeout (120s, see ``download_image_bytes``)
plus the default poll timeout margin. 60s would expire while a permitted
download is still inside its own 120s timeout. The shared upstream queue
(100 slots / 30s) is a different process and is not modified here.

``active + waiting`` counts large jobs only. 1K generations, health, and
task polling do not take a slot. While 5 heavy stages are running, further
large jobs hold a waiting slot and do not call upstream until a heavy
stage finishes or the wait times out. That is intentionally narrower than
"at most 5 large generations ever", and it is not Sub2 group #44.
"""

from __future__ import annotations

import base64
import importlib.util
import logging
import os
import threading
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger("chatgpt2api.free_large_image")

# Pixels above this are large. 1024*1536 and 1254*1254 stay at or under it.
# 2048*2048, 3840*2160, and 4096*4096 are above it. Equality is still small.
LARGE_PIXEL_THRESHOLD = 1_700_000
MAX_ACTIVE = 5
MAX_WAITING = 10
# 120s download timeout + 60s for one wave of decode/upscale/base64.
WAIT_TIMEOUT_SECS = 180
# Header sniff budget. Stops well before a full 2K/4K buffer is assembled.
SNIFF_LIMIT = 256 * 1024
# Only when dimensions cannot be read. Ordinary 1K payloads are smaller.
UNKNOWN_LARGE_BYTES = 8_000_000

_SIZE_ALIASES = {
    "1k": (1024, 1024),
    "2k": (2048, 2048),
    "4k": (4096, 4096),
}
_OFF = {"0", "false", "off", "no", "disabled"}
_ON = {"1", "true", "on", "yes", "free"}

_tls = threading.local()


class LargeImageAdmissionError(Exception):
    """Raised before a new upstream generation, or when heavy entry times out.

    ``reason`` is one of ``queue_full``, ``queue_timeout``, ``cancelled``.
    The message is metadata only. Callers must not start a replacement
    generation for the same attempt.
    """

    def __init__(self, reason: str) -> None:
        self.reason = str(reason or "queue_full")
        super().__init__(self.reason)


class _Guard:
    def __init__(self) -> None:
        self.cv = threading.Condition()
        self.enabled = False
        self.configured = False
        self.max_active = MAX_ACTIVE
        self.max_waiting = MAX_WAITING
        self.wait_timeout_secs = WAIT_TIMEOUT_SECS
        self.threshold = LARGE_PIXEL_THRESHOLD
        self.active = 0
        self.waiting = 0
        self.rejected = 0
        self.timed_out = 0
        self.cancelled = 0
        self.peak_active = 0
        self.peak_waiting = 0
        self._next_id = 1
        self.tokens: dict[int, Token] = {}
        self.gen_queue: deque[Token] = deque()
        self.heavy_queue: deque[Token] = deque()

    def stats(self) -> dict[str, Any]:
        with self.cv:
            return {
                "enabled": self.enabled,
                "scope": "process",
                "cross_process": False,
                "active": self.active,
                "waiting": self.waiting,
                "max_active": self.max_active,
                "max_waiting": self.max_waiting,
                "wait_timeout_secs": self.wait_timeout_secs,
                "threshold_pixels": self.threshold,
                "rejected": self.rejected,
                "timed_out": self.timed_out,
                "cancelled": self.cancelled,
                "peak_active": self.peak_active,
                "peak_waiting": self.peak_waiting,
            }

    def reset(self, *, enabled: bool) -> None:
        with self.cv:
            self.enabled = bool(enabled)
            self.configured = True
            self.active = 0
            self.waiting = 0
            self.rejected = 0
            self.timed_out = 0
            self.cancelled = 0
            self.peak_active = 0
            self.peak_waiting = 0
            self.tokens.clear()
            self.gen_queue.clear()
            self.heavy_queue.clear()
            self.cv.notify_all()


_guard = _Guard()


class Token:
    def __init__(self, guard: _Guard) -> None:
        self.guard = guard
        self.id = 0
        self.state = "new"  # queued | reserved | active | done
        self.cancelled = False
        self.closed = False
        self.work_running = False
        self.depth = 0
        self.stage = ""

    def cancel(self) -> bool:
        """Cancel a job that has not entered heavy work. Never drops an active permit."""
        guard = self.guard
        with guard.cv:
            if self.work_running or self.state == "active":
                return False
            if self.state == "done":
                return False
            self.cancelled = True
            guard.cv.notify_all()
            return True

    def close(self) -> None:
        """Release a waiting slot. A running heavy section keeps its permit."""
        guard = self.guard
        with guard.cv:
            if self.state == "done" and not self.work_running:
                guard.tokens.pop(self.id, None)
                return
            self.closed = True
            if self.work_running or self.state == "active":
                return
            self._discard_waiting_locked()
            self.state = "done"
            guard.tokens.pop(self.id, None)
            guard.cv.notify_all()

    def _discard_waiting_locked(self) -> None:
        guard = self.guard
        if self in guard.gen_queue:
            guard.gen_queue.remove(self)
        if self in guard.heavy_queue:
            guard.heavy_queue.remove(self)
        if self.state in {"queued", "reserved"} and guard.waiting > 0:
            guard.waiting -= 1
        elif self.state in {"queued", "reserved"}:
            guard.waiting = 0

    def wait_generation_allowed(self, timeout: float) -> None:
        """Block before upstream generation while every heavy slot is busy."""
        guard = self.guard
        deadline = time.monotonic() + max(0.0, timeout)
        with guard.cv:
            while self.state == "queued":
                if self.cancelled:
                    guard.cancelled += 1
                    raise LargeImageAdmissionError("cancelled")
                head_ok = not guard.gen_queue or guard.gen_queue[0] is self
                if guard.active < guard.max_active and head_ok:
                    if guard.gen_queue and guard.gen_queue[0] is self:
                        guard.gen_queue.popleft()
                    self.state = "reserved"
                    guard.cv.notify_all()
                    return
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    guard.timed_out += 1
                    logger.warning(
                        "free_large_image admission timeout active=%s waiting=%s",
                        guard.active,
                        guard.waiting,
                    )
                    raise LargeImageAdmissionError("queue_timeout")
                guard.cv.wait(remaining)

    def enter_heavy(self, stage: str, timeout: float) -> None:
        guard = self.guard
        with guard.cv:
            if self.state == "active" and self.work_running:
                self.depth += 1
                return
            if self.cancelled or self.state != "reserved":
                guard.cancelled += 1
                raise LargeImageAdmissionError("cancelled")
            guard.heavy_queue.append(self)
            deadline = time.monotonic() + max(0.0, timeout)
            try:
                while True:
                    if self.cancelled:
                        guard.cancelled += 1
                        raise LargeImageAdmissionError("cancelled")
                    head_ok = bool(guard.heavy_queue) and guard.heavy_queue[0] is self
                    if guard.active < guard.max_active and head_ok:
                        guard.heavy_queue.popleft()
                        break
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        guard.timed_out += 1
                        logger.warning(
                            "free_large_image heavy-entry timeout active=%s waiting=%s stage=%s",
                            guard.active,
                            guard.waiting,
                            stage,
                        )
                        raise LargeImageAdmissionError("queue_timeout")
                    guard.cv.wait(remaining)
            except BaseException:
                if self in guard.heavy_queue:
                    guard.heavy_queue.remove(self)
                raise
            if guard.waiting > 0:
                guard.waiting -= 1
            guard.active += 1
            if guard.active > guard.peak_active:
                guard.peak_active = guard.active
            self.state = "active"
            self.work_running = True
            self.depth = 1
            self.stage = stage
            guard.cv.notify_all()

    def leave_heavy(self) -> None:
        guard = self.guard
        with guard.cv:
            if self.depth > 1:
                self.depth -= 1
                return
            if self.state != "active":
                self.depth = 0
                self.work_running = False
                return
            self.depth = 0
            self.work_running = False
            if guard.active > 0:
                guard.active -= 1
            if self.closed:
                self.state = "done"
                guard.tokens.pop(self.id, None)
            else:
                self.state = "reserved"
                guard.waiting += 1
                if guard.waiting > guard.peak_waiting:
                    guard.peak_waiting = guard.waiting
            guard.cv.notify_all()


def _env_flag() -> str:
    return os.environ.get("CHATGPT2API_FREE_LARGE_IMAGE_GUARD", "").strip().lower()


def _free_pool_present() -> bool:
    try:
        return importlib.util.find_spec("services.free_pool_policy") is not None
    except Exception:
        return False


def _autoload() -> None:
    if _guard.configured:
        return
    flag = _env_flag()
    if flag in _OFF:
        _guard.enabled = False
    elif flag in _ON:
        _guard.enabled = True
    else:
        # Free deployments mount this package. Plus and Adobe do not.
        _guard.enabled = _free_pool_present()
    _guard.configured = True


def is_enabled() -> bool:
    _autoload()
    return bool(_guard.enabled)


def install_free_large_image_guard(enabled: bool = True) -> None:
    """Turn the guard on or off. An explicit off environment variable wins."""
    if _env_flag() in _OFF:
        enabled = False
    _guard.enabled = bool(enabled)
    _guard.configured = True


def configure_limits(
    *,
    max_active: int | None = None,
    max_waiting: int | None = None,
    wait_timeout_secs: float | None = None,
    threshold_pixels: int | None = None,
) -> None:
    with _guard.cv:
        if max_active is not None:
            _guard.max_active = max(1, int(max_active))
        if max_waiting is not None:
            _guard.max_waiting = max(0, int(max_waiting))
        if wait_timeout_secs is not None:
            _guard.wait_timeout_secs = max(0.0, float(wait_timeout_secs))
        if threshold_pixels is not None:
            _guard.threshold = max(1, int(threshold_pixels))


def reset_for_tests(*, enabled: bool = False) -> None:
    _guard.reset(enabled=enabled)
    _guard.max_active = MAX_ACTIVE
    _guard.max_waiting = MAX_WAITING
    _guard.wait_timeout_secs = WAIT_TIMEOUT_SECS
    _guard.threshold = LARGE_PIXEL_THRESHOLD
    stack = getattr(_tls, "stack", None)
    if stack is not None:
        stack.clear()
    pipelines = getattr(_tls, "pipelines", None)
    if pipelines is not None:
        pipelines.clear()


def stats() -> dict[str, Any]:
    _autoload()
    return _guard.stats()


def cancel_token(token_id: int) -> bool:
    with _guard.cv:
        token = _guard.tokens.get(int(token_id))
    if token is None:
        return False
    return token.cancel()


def token_snapshot() -> list[dict[str, Any]]:
    """Ids and states only. No prompts, keys, URLs, or image bytes."""
    with _guard.cv:
        return [
            {"id": token.id, "state": token.state, "stage": token.stage}
            for token in _guard.tokens.values()
        ]


def resolve_dimensions(size: object) -> tuple[int, int] | None:
    """Parse a requested size. ``None`` means unknown, not a silent 1024 square.

    Accepts ``1K`` / ``2K`` / ``4K`` and explicit ``WxH`` (including ``×``).
    Does not apply a default. Malformed input is unknown and therefore not large.
    """
    if isinstance(size, (tuple, list)) and len(size) >= 2:
        try:
            width = int(size[0])
            height = int(size[1])
        except (TypeError, ValueError):
            return None
        if width > 0 and height > 0:
            return width, height
        return None
    text = str(size or "").strip()
    if not text:
        return None
    lower = text.lower()
    if lower in {"auto", "default", "none", "null"}:
        return None
    if lower in _SIZE_ALIASES:
        return _SIZE_ALIASES[lower]
    import re

    match = re.match(r"^\s*(\d{2,5})\s*[xX×]\s*(\d{2,5})\s*$", text)
    if match is None:
        match = re.search(r"(\d{2,5})\s*[xX×]\s*(\d{2,5})", text)
    if match is None:
        return None
    width, height = int(match.group(1)), int(match.group(2))
    if width <= 0 or height <= 0:
        return None
    return width, height


def pixel_count(size: object) -> int | None:
    dims = resolve_dimensions(size)
    if dims is None:
        return None
    return int(dims[0]) * int(dims[1])


def is_large_pixel_count(pixels: int | None) -> bool:
    if pixels is None or pixels <= 0:
        return False
    _autoload()
    return int(pixels) > int(_guard.threshold)


def is_large_size(size: object) -> bool:
    return is_large_pixel_count(pixel_count(size))


def sniff_image_size(data: bytes) -> tuple[int, int] | None:
    """Read width and height from header bytes. Does not decode pixels."""
    if not data:
        return None
    try:
        if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24 and data[12:16] == b"IHDR":
            width = int.from_bytes(data[16:20], "big")
            height = int.from_bytes(data[20:24], "big")
            return (width, height) if width > 0 and height > 0 else None
        if data.startswith((b"GIF87a", b"GIF89a")) and len(data) >= 10:
            width = int.from_bytes(data[6:8], "little")
            height = int.from_bytes(data[8:10], "little")
            return (width, height) if width > 0 and height > 0 else None
        if data.startswith(b"RIFF") and len(data) >= 30 and data[8:12] == b"WEBP":
            return _sniff_webp(data)
        if data.startswith(b"\xff\xd8"):
            return _sniff_jpeg(data)
    except Exception:
        return None
    return None


def _sniff_webp(data: bytes) -> tuple[int, int] | None:
    chunk = data[12:16]
    if chunk == b"VP8X" and len(data) >= 30:
        width = 1 + int.from_bytes(data[24:27], "little")
        height = 1 + int.from_bytes(data[27:30], "little")
        return (width, height) if width > 0 and height > 0 else None
    if chunk == b"VP8 " and len(data) >= 30:
        # Lossy bitstream: 3-byte frame tag after the chunk header, then width/height.
        start = 20
        if data[start:start + 3] != b"\x9d\x01\x2a":
            start = data.find(b"\x9d\x01\x2a", 20, 64)
            if start < 0:
                return None
        width = int.from_bytes(data[start + 3:start + 5], "little") & 0x3FFF
        height = int.from_bytes(data[start + 5:start + 7], "little") & 0x3FFF
        return (width, height) if width > 0 and height > 0 else None
    if chunk == b"VP8L" and len(data) >= 25 and data[20] == 0x2F:
        bits = int.from_bytes(data[21:25], "little")
        width = (bits & 0x3FFF) + 1
        height = ((bits >> 14) & 0x3FFF) + 1
        return width, height
    return None


def _sniff_jpeg(data: bytes) -> tuple[int, int] | None:
    sof = {
        0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
        0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
    }
    index = 2
    limit = min(len(data), SNIFF_LIMIT)
    while index + 8 < limit:
        if data[index] != 0xFF:
            index += 1
            continue
        while index < limit and data[index] == 0xFF:
            index += 1
        if index >= limit:
            return None
        marker = data[index]
        index += 1
        if marker == 0xD8:
            continue
        if marker == 0xD9 or marker == 0xDA:
            return None
        if index + 1 >= limit:
            return None
        segment = int.from_bytes(data[index:index + 2], "big")
        if segment < 2:
            return None
        if marker in sof and index + 7 <= limit:
            height = int.from_bytes(data[index + 3:index + 5], "big")
            width = int.from_bytes(data[index + 5:index + 7], "big")
            return (width, height) if width > 0 and height > 0 else None
        index += segment
    return None


def image_bytes_are_large(data: bytes) -> bool:
    dims = sniff_image_size(data[:SNIFF_LIMIT])
    if dims is None:
        return False
    return is_large_pixel_count(dims[0] * dims[1])


def image_inputs_are_large(images: object) -> bool:
    """Header-only check of already buffered edit inputs. Does not decode them."""
    if not images:
        return False
    entries = images if isinstance(images, (list, tuple)) else [images]
    for item in entries:
        raw = item[0] if isinstance(item, tuple) and item else item
        if isinstance(raw, (bytes, bytearray)) and image_bytes_are_large(bytes(raw)):
            return True
    return False


def max_input_pixels(images: object) -> int:
    best = 0
    if not images:
        return 0
    entries = images if isinstance(images, (list, tuple)) else [images]
    for item in entries:
        raw = item[0] if isinstance(item, tuple) and item else item
        if not isinstance(raw, (bytes, bytearray)):
            continue
        dims = sniff_image_size(bytes(raw)[:SNIFF_LIMIT])
        if dims:
            best = max(best, int(dims[0]) * int(dims[1]))
    return best


def b64_sample_pixels(b64_text: str) -> int | None:
    """Decode only a header-sized prefix of a base64 payload."""
    text = str(b64_text or "").strip()
    if text.startswith("data:") and "," in text:
        text = text.split(",", 1)[1]
    sample = text[: ((SNIFF_LIMIT + 2) // 3) * 4]
    if len(sample) < 16:
        return None
    pad = (-len(sample)) % 4
    try:
        data = base64.b64decode(sample + ("=" * pad), validate=False)
    except Exception:
        return None
    dims = sniff_image_size(data)
    if dims is None:
        return None
    return int(dims[0]) * int(dims[1])


def _current_token() -> Token | None:
    stack = getattr(_tls, "stack", None)
    if not stack:
        return None
    token = stack[-1]
    if token.state == "done" or token.closed and not token.work_running:
        return None
    return token


def _push_token(token: Token) -> None:
    stack = getattr(_tls, "stack", None)
    if stack is None:
        stack = []
        _tls.stack = stack
    stack.append(token)


def _pop_token(token: Token) -> None:
    stack = getattr(_tls, "stack", None)
    if not stack:
        return
    if stack[-1] is token:
        stack.pop()
    elif token in stack:
        stack.remove(token)


def _acquire_before_generation() -> Token:
    timeout = _guard.wait_timeout_secs
    with _guard.cv:
        if _guard.waiting >= _guard.max_waiting:
            _guard.rejected += 1
            logger.warning(
                "free_large_image queue full active=%s waiting=%s max_waiting=%s",
                _guard.active,
                _guard.waiting,
                _guard.max_waiting,
            )
            raise LargeImageAdmissionError("queue_full")
        _guard.waiting += 1
        if _guard.waiting > _guard.peak_waiting:
            _guard.peak_waiting = _guard.waiting
        token = Token(_guard)
        token.id = _guard._next_id
        _guard._next_id += 1
        token.state = "queued"
        _guard.tokens[token.id] = token
        _guard.gen_queue.append(token)
    try:
        token.wait_generation_allowed(timeout)
    except BaseException:
        token.close()
        raise
    return token


@contextmanager
def bind_large_image_job(size: object) -> Iterator[Token | None]:
    """Reserve a large-output job before upstream generation.

    No-op for small sizes and when the guard is disabled. When every heavy
    slot is busy, this waits (bounded) and only then lets the caller start
    upstream work. Queue overflow and timeout raise ``LargeImageAdmissionError``
    without yielding.
    """
    if not is_enabled() or not is_large_size(size):
        yield None
        return
    current = _current_token()
    if current is not None and current.state in {"queued", "reserved", "active"}:
        yield current
        return
    token = _acquire_before_generation()
    _push_token(token)
    try:
        yield token
    finally:
        _pop_token(token)
        token.close()


class _HeavyPipeline:
    """One permit owned by the caller until every heavy phase on this thread ends.

    A download may discover that the bytes are large after admission looked at
    a small or missing ``size``. ``cover`` takes the permit at that moment.
    Leaving the download helper must not drop it; only ``release`` does.
    """

    def __init__(self, declared_size: object, stage: str) -> None:
        self.declared_size = declared_size
        self.stage = stage
        self.token: Token | None = None
        self.owned = False
        self.entered = False
        self.discovered_pixels = 0

    def cover(self, pixels: int | None = None, stage: str | None = None) -> None:
        if not is_enabled():
            return
        if pixels:
            self.discovered_pixels = max(self.discovered_pixels, int(pixels))
        large = is_large_size(self.declared_size) or is_large_pixel_count(self.discovered_pixels)
        if not large or self.entered:
            return
        if self.token is None:
            current = _current_token()
            if current is None or current.state not in {"queued", "reserved", "active"}:
                current = _acquire_before_generation()
                self.owned = True
                _push_token(current)
            self.token = current
        if self.token.state != "active":
            self.token.enter_heavy(stage or self.stage, _guard.wait_timeout_secs)
            self.entered = True
        else:
            # Already inside this thread's heavy section. Do not take a second slot.
            self.entered = False

    def release(self) -> None:
        token = self.token
        try:
            if self.entered and token is not None:
                token.leave_heavy()
        finally:
            self.entered = False
            if self.owned and token is not None:
                _pop_token(token)
                token.close()
            self.owned = False
            self.token = None


def _current_pipeline() -> _HeavyPipeline | None:
    stack = getattr(_tls, "pipelines", None)
    if not stack:
        return None
    return stack[-1]


def _push_pipeline(pipeline: _HeavyPipeline) -> None:
    stack = getattr(_tls, "pipelines", None)
    if stack is None:
        stack = []
        _tls.pipelines = stack
    stack.append(pipeline)


def _pop_pipeline(pipeline: _HeavyPipeline) -> None:
    stack = getattr(_tls, "pipelines", None)
    if not stack:
        return
    if stack[-1] is pipeline:
        stack.pop()
    elif pipeline in stack:
        stack.remove(pipeline)


@contextmanager
def heavy_image_pipeline(
    declared_size: object = None,
    stage: str = "download_decode",
) -> Iterator[_HeavyPipeline | None]:
    """Keep one permit from the first large discovery through later heavy phases.

    Declared 2K/4K takes the permit before the body. A 1K, ``auto``, or missing
    size takes it when the download (or another phase) sees a large pixel
    count, and that same permit still covers decode, upscale, and base64.
    """
    if not is_enabled():
        yield None
        return
    pipeline = _HeavyPipeline(declared_size, stage)
    _push_pipeline(pipeline)
    try:
        if is_large_size(declared_size):
            pipeline.cover(pixel_count(declared_size), stage=stage)
        yield pipeline
    finally:
        try:
            pipeline.release()
        finally:
            _pop_pipeline(pipeline)


@contextmanager
def heavy_image_work(
    declared_size: object = None,
    discovered_pixels: int | None = None,
    stage: str = "heavy",
) -> Iterator[Token | None]:
    """Hold one heavy-memory permit around the caller's allocation.

    Nested use on the same thread reuses the permit (no deadlock, no second
    slot). Inside ``heavy_image_pipeline`` a discovery extends that pipeline
    instead of opening a permit that would be released when this block ends.
    The permit stays held if another thread asks to release it early.
    """
    if not is_enabled():
        yield None
        return
    pixels = int(discovered_pixels or 0)
    large = is_large_size(declared_size) or is_large_pixel_count(pixels)
    pipeline = _current_pipeline()
    if pipeline is not None:
        if large:
            pipeline.cover(pixels or pixel_count(declared_size), stage=stage)
        yield pipeline.token
        return
    if not large:
        yield None
        return
    token = _current_token()
    owned = False
    if token is None or token.state not in {"queued", "reserved", "active"}:
        token = _acquire_before_generation()
        owned = True
        _push_token(token)
    try:
        token.enter_heavy(stage, _guard.wait_timeout_secs)
        try:
            yield token
        finally:
            token.leave_heavy()
    finally:
        if owned:
            _pop_token(token)
            token.close()


def read_stream_guarded(
    chunks: Iterator[bytes],
    *,
    declared_size: object = None,
    content_length: int | None = None,
    stage: str = "download",
) -> bytes:
    """Join a download, acquiring a permit before the remainder of a large body.

    A declared large size acquires before the first body byte. Otherwise only
    a sniff buffer is kept until dimensions (or the sniff limit) are known.
    """
    if is_enabled() and is_large_size(declared_size):
        with heavy_image_work(declared_size=declared_size, stage=stage):
            return _join_chunks(chunks)

    prefix = bytearray()
    for chunk in chunks:
        if not chunk:
            continue
        prefix.extend(chunk)
        ready, dims = _sniff_ready(prefix)
        if not ready:
            if len(prefix) > SNIFF_LIMIT:
                break
            continue
        large = _discovered_large(dims, content_length)
        if large:
            pixels = dims[0] * dims[1] if dims else _guard.threshold + 1
            rest = [bytes(prefix)]
            with heavy_image_work(discovered_pixels=pixels, stage=stage):
                rest.extend(item for item in chunks if item)
                return b"".join(rest)
        break

    parts = [bytes(prefix)]
    parts.extend(item for item in chunks if item)
    return b"".join(parts)


def _sniff_ready(buf: bytes | bytearray) -> tuple[bool, tuple[int, int] | None]:
    data = bytes(buf)
    dims = sniff_image_size(data)
    if dims is not None:
        return True, dims
    if data.startswith(b"\xff\xd8") and len(data) < SNIFF_LIMIT:
        return False, None
    if len(data) >= 32 or len(data) >= SNIFF_LIMIT:
        return True, None
    return False, None


def _discovered_large(dims: tuple[int, int] | None, content_length: int | None) -> bool:
    if dims is not None:
        return is_large_pixel_count(dims[0] * dims[1])
    if content_length is not None and int(content_length) >= UNKNOWN_LARGE_BYTES:
        return True
    return False


def _join_chunks(chunks: Iterator[bytes]) -> bytes:
    return b"".join(chunk for chunk in chunks if chunk)


def iter_response_chunks(response: Any, chunk_size: int = 65536) -> Iterator[bytes]:
    iterator = getattr(response, "iter_content", None)
    if callable(iterator):
        yield from iterator(chunk_size=chunk_size)
        return
    content = getattr(response, "content", b"") or b""
    if content:
        yield content


def guarded_response_bytes(
    response: Any,
    *,
    declared_size: object = None,
    stage: str = "download",
) -> bytes:
    length = _content_length(response)
    return read_stream_guarded(
        iter_response_chunks(response),
        declared_size=declared_size,
        content_length=length,
        stage=stage,
    )


def _content_length(response: Any) -> int | None:
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    raw = headers.get("Content-Length") if hasattr(headers, "get") else None
    if raw is None:
        return None
    text = str(raw).strip()
    if text.isdigit():
        return int(text)
    return None


def postprocess_image_bytes(raw: bytes, size: object) -> bytes:
    """Optional free-pool Lanczos hook. Missing module leaves bytes unchanged.

    The shared Plus/Adobe size proxy is a different program and is not called.
    """
    if not raw or size is None:
        return raw
    try:
        from utils.fake_upscale import maybe_upscale_image_bytes
    except Exception:
        return raw
    try:
        processed = maybe_upscale_image_bytes(raw, size)
    except Exception:
        logger.warning("free_large_image upscale failed stage=postprocess")
        return raw
    return raw if processed is None else processed
