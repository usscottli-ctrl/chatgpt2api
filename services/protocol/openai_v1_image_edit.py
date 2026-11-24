from __future__ import annotations

import threading
from io import BytesIO
from typing import Any, Iterator

from PIL import Image

from services.protocol.conversation import (
    ConversationRequest,
    ImageGenerationError,
    collect_image_outputs,
    count_text_tokens,
    encode_images,
    stream_image_chunks,
    stream_image_outputs_with_pool,
)
from utils.image_tokens import count_image_inputs_tokens, count_image_output_items_tokens, image_usage


class _CappedBytesIO(BytesIO):
    """PNG writer that cannot grow past the reservation."""

    def __init__(self, limit: int) -> None:
        super().__init__()
        self._limit = int(limit)

    def write(self, data: bytes) -> int:
        incoming = data
        if self.tell() + len(incoming) > self._limit:
            self.seek(0)
            self.truncate(0)
            self.close()
            incoming = None
            data = None
            from services.free_large_image_guard import LargeImageAdmissionError

            raise LargeImageAdmissionError("buffer_budget")
        return super().write(incoming)

    def getvalue(self) -> bytes:
        if self.closed:
            return b""
        return super().getvalue()


class _EditStream:
    """One edit stream and the buffer lease it owns.

    ``next`` is single-threaded. A second caller is rejected and does not
    enter the factory. ``close`` is idempotent from any thread: it marks the
    lease cancelled, and a native step already inside ``next`` keeps the
    bytes until it returns. That return does not start upstream.

    Exhaustion, a generator error, and a cancelled HTTP request all clear
    the payload before ``next`` returns, even if the caller keeps this
    wrapper. Garbage collection uses the same ``close``.
    """

    def __init__(self, factory, hold, cleanup) -> None:
        self._factory = factory
        self._hold = hold
        self._cleanup = cleanup
        self._gen = None
        self._closed = False
        self._gate = threading.Lock()
        self._state = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        if not self._gate.acquire(blocking=False):
            raise RuntimeError("edit stream is already advancing")
        try:
            if self._closed:
                raise StopIteration
            self._hold.enter_native()
            terminal = False
            try:
                if self._gen is None:
                    self._gen = self._factory()
                item = next(self._gen)
            except StopIteration:
                terminal = True
                raise
            except BaseException:
                terminal = True
                raise
            else:
                return item
            finally:
                stopped = self._hold.leave_native()
                if terminal or stopped or _stream_request_cancelled():
                    self._finish_stream()
        finally:
            self._gate.release()

    def _finish_stream(self) -> None:
        """Clear payload once this stream cannot produce another upstream call."""
        self._closed = True
        self._hold.request_close(self._cleanup)
        self._drop_generator()

    def _drop_generator(self) -> None:
        """Drop a suspended generator. Never call this while next() is inside it."""
        with self._state:
            if self._hold.native:
                return
            gen = self._gen
            self._gen = None
        if gen is None:
            return
        closer = getattr(gen, "close", None)
        if closer is None:
            return
        try:
            closer()
        except Exception:
            pass

    def close(self) -> None:
        self._closed = True
        self._hold.request_close(self._cleanup)
        self._drop_generator()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _stream_request_cancelled() -> bool:
    from services.free_large_image_guard import request_cancel_is_set

    return request_cancel_is_set()


def _composite_mask(
    images: list[tuple[bytes, str, str]],
    masks: list[tuple[bytes, str, str]],
    limits: list[int] | None = None,
) -> list[tuple[bytes, str, str]]:
    """将 mask 的 alpha 通道合成到图片中，标识需要编辑的区域。
    
    mask 的透明区域（低 alpha）= 需要编辑的区域，
    mask 的不透明区域（高 alpha）= 保留的区域。
    如果无 mask 则返回原图。
    """
    if not masks:
        return images
    result: list[tuple[bytes, str, str]] = []
    try:
        return _composite_mask_into(images, masks, limits, result)
    except BaseException:
        result.clear()
        raise


def _close_quiet(obj: object) -> None:
    closer = getattr(obj, "close", None)
    if closer is None:
        return
    try:
        closer()
    except Exception:
        pass


def _composite_mask_into(
    images: list[tuple[bytes, str, str]],
    masks: list[tuple[bytes, str, str]],
    limits: list[int] | None,
    result: list[tuple[bytes, str, str]],
) -> list[tuple[bytes, str, str]]:
    """Composite each image. Failure drops this frame's upload and PNG buffer.

    The exception can stay on the caller's traceback. The bytes it must not
    keep are the source ``data`` local and the writer ``buf``. Those are
    cleared here, before the exception leaves this frame. ``result`` is the
    caller's list; the caller clears it when this function raises.
    """
    data = None
    mask_data = None
    filename = None
    mime_type = None
    source = None
    mask_source = None
    img = None
    mask_img = None
    converted = None
    alpha = None
    icc = None
    buf: BytesIO | None = None
    payload = None
    try:
        for i, (data, filename, mime_type) in enumerate(images):
            mask_data = masks[i][0] if i < len(masks) else masks[-1][0]
            source = BytesIO(data)
            try:
                img = Image.open(source)
                loader = getattr(img, "load", None)
                if loader is not None:
                    loader()
                converted = img.convert("RGBA")
                if converted is not img:
                    _close_quiet(img)
                img = converted
                converted = None
            finally:
                source.close()
                source = None
            mask_source = BytesIO(mask_data)
            try:
                mask_img = Image.open(mask_source)
                loader = getattr(mask_img, "load", None)
                if loader is not None:
                    loader()
            finally:
                mask_source.close()
                mask_source = None
            if mask_img.mode == "RGBA":
                alpha = mask_img.split()[3]
            elif mask_img.mode == "L":
                alpha = mask_img
            else:
                alpha = mask_img.convert("L")
            alpha = alpha.resize(img.size, Image.LANCZOS)
            img.putalpha(alpha)
            # Pillow's PNG writer copies icc_profile. Dropping it would change
            # color. ``limits[i]`` is the reservation for that profile.
            info = img.info if isinstance(getattr(img, "info", None), dict) else {}
            icc = info.get("icc_profile")
            if limits is not None:
                buf = _CappedBytesIO(int(limits[i] if i < len(limits) else 0))
            else:
                buf = BytesIO()
            if isinstance(icc, (bytes, bytearray)) and icc:
                img.save(buf, format="PNG", icc_profile=icc)
            else:
                img.save(buf, format="PNG")
            payload = buf.getvalue()
            buf.close()
            buf = None
            result.append((payload, filename, "image/png"))
            payload = None
            icc = None
            _close_quiet(img)
            _close_quiet(mask_img)
            img = None
            mask_img = None
            alpha = None
            data = None
            mask_data = None
        return result
    finally:
        _close_quiet(buf)
        buf = None
        payload = None
        icc = None
        _close_quiet(source)
        _close_quiet(mask_source)
        source = None
        mask_source = None
        _close_quiet(img)
        _close_quiet(converted)
        _close_quiet(mask_img)
        img = None
        converted = None
        mask_img = None
        alpha = None
        data = None
        mask_data = None
        filename = None
        mime_type = None


def _encode_retained(images: list) -> list[str]:
    """Encode, then drop any source byte local this frame still names."""
    data = None
    try:
        return encode_images(images)
    finally:
        data = None


def handle(body: dict[str, Any]) -> dict[str, Any] | Iterator[dict[str, Any]]:
    prompt = str(body.get("prompt") or "")
    images = body.get("images") or []
    masks = body.get("mask") or []
    from services.free_large_image_guard import (
        LargeImageAdmissionError,
        base64_payload_length,
        bind_large_image_job,
        composite_output_caps,
        edit_input_guard_pixels,
        heavy_image_work,
        icc_chunk_upper_bound,
        is_enabled,
        payload_byte_length,
        png_rgba_upper_bound,
        raise_if_request_cancelled,
        reserve_edit_buffers,
    )
    from services.protocol.conversation import _admission_error_code

    guard_pixels = edit_input_guard_pixels(images, masks)
    hold = None

    def _prepare() -> None:
        nonlocal images, encoded_images
        produced = None
        item = None
        part = None
        profile = None
        try:
            if guard_pixels:
                with heavy_image_work(discovered_pixels=guard_pixels, stage="edit_input"):
                    if masks and hold is not None:
                        caps = composite_output_caps(images)
                        if caps is None:
                            caps = []
                            for item in images:
                                part = item[0] if isinstance(item, tuple) and item else b""
                                with Image.open(BytesIO(part)) as opened:
                                    opened.load()
                                    info = opened.info if isinstance(getattr(opened, "info", None), dict) else {}
                                    profile = info.get("icc_profile")
                                    extra = icc_chunk_upper_bound(len(profile)) if isinstance(profile, (bytes, bytearray)) else 0
                                    profile = None
                                    caps.append(png_rgba_upper_bound(opened.size[0], opened.size[1], extra))
                            item = None
                            part = None
                            profile = None
                        upper = sum(caps)
                        hold.add(upper)
                        produced = _composite_mask(images, masks, caps)
                        actual = payload_byte_length(produced)
                        if actual > upper:
                            produced.clear()
                            produced = None
                            raise LargeImageAdmissionError("buffer_budget")
                        hold.release_amount(upper - actual)
                        images = produced
                        produced = None
                    else:
                        images = _composite_mask(images, masks)
                    raise_if_request_cancelled()
                    if hold is not None:
                        encoded_need = 0
                        for item in images or []:
                            part = item[0] if isinstance(item, tuple) and item else b""
                            if part:
                                encoded_need += base64_payload_length(len(part))
                        item = None
                        part = None
                        hold.add(encoded_need)
                    encoded_images = _encode_retained(images)
                raise_if_request_cancelled()
            else:
                images = _composite_mask(images, masks)
                encoded_images = _encode_retained(images)
        finally:
            if isinstance(produced, list):
                produced.clear()
            produced = None
            item = None
            part = None
            profile = None

    encoded_images = None
    model = str(body.get("model") or "gpt-image-2")
    n = int(body.get("n") or 1)
    size = body.get("size")
    quality = str(body.get("quality") or "auto")
    response_format = str(body.get("response_format") or "b64_json")
    base_url = str(body.get("base_url") or "") or None
    progress_callback = body.get("progress_callback")

    def _finish(outputs):
        if body.get("stream"):
            return stream_image_chunks(outputs)
        result = collect_image_outputs(outputs)
        result["usage"] = image_usage(
            input_text_tokens=count_text_tokens(prompt, model),
            input_image_tokens=count_image_inputs_tokens(images, model),
            output_tokens=count_image_output_items_tokens(result.get("data"), size, quality),
        )
        return result

    def _request() -> ConversationRequest:
        return ConversationRequest(
            prompt=prompt,
            model=model,
            n=n,
            size=size,
            quality=quality,
            response_format=response_format,
            base_url=base_url,
            images=encoded_images,
            message_as_error=True,
            progress_callback=progress_callback,
        )

    # Uploads are already in memory before this function. A large edit reserves
    # those bytes before it waits. Decode and base64 are reserved immediately
    # before they are created, and only after a claim is held, so a child that
    # is still waiting does not keep an encoded copy. n=1 non-stream keeps one
    # claim through generation. n>1 and stream encode once inside the first
    # child claim so the parent claim cannot deadlock the children.
    def _wipe(value: object) -> None:
        if isinstance(value, list):
            value.clear()

    def _clear_payload() -> None:
        nonlocal images, encoded_images, masks
        # Mutate the list objects frames may already point at. Rebinding the
        # name leaves the previous list reachable from an exception traceback.
        _wipe(images)
        _wipe(masks)
        _wipe(encoded_images)
        _wipe(body.get("images"))
        _wipe(body.get("mask"))
        images = []
        masks = []
        encoded_images = None
        body["images"] = []
        body["mask"] = []

    def _busy(exc: LargeImageAdmissionError) -> ImageGenerationError:
        return ImageGenerationError(
            "Free-pool large-image capacity is busy. Retry this same request later; "
            "do not start a duplicate generation.",
            status_code=429,
            error_type="rate_limit_error",
            code=_admission_error_code(exc.reason),
        )

    if not (guard_pixels and is_enabled()):
        _prepare()
        if not encoded_images:
            raise ImageGenerationError("image is required")
        return _finish(stream_image_outputs_with_pool(_request()))

    try:
        hold = reserve_edit_buffers(payload_byte_length(images) + payload_byte_length(masks))
    except LargeImageAdmissionError as exc:
        _clear_payload()
        raise _busy(exc) from exc

    def _release() -> None:
        _clear_payload()
        if hold is not None:
            hold.release()

    handed_off = False
    try:
        if n == 1 and not body.get("stream"):
            with bind_large_image_job("4K"):
                _prepare()
                if not encoded_images:
                    raise ImageGenerationError("image is required")
                return _finish(stream_image_outputs_with_pool(_request()))

        gate = threading.Lock()
        state: dict[str, Any] = {"done": False, "error": None}

        def _prepare_once(request: ConversationRequest) -> None:
            with gate:
                if state["error"] is not None:
                    raise state["error"]
                if hold is not None and hold.cancel:
                    raise LargeImageAdmissionError("cancelled")
                if not state["done"]:
                    try:
                        _prepare()
                        state["done"] = True
                    except BaseException as exc:
                        state["error"] = exc
                        raise
                if hold is not None and hold.cancel:
                    raise LargeImageAdmissionError("cancelled")
                # Same list the lease clears. A copy would stay reachable from
                # the request on an exception traceback.
                request.images = encoded_images if isinstance(encoded_images, list) else []

        def _outputs():
            request = _request()
            request.before_upstream = _prepare_once
            return stream_image_outputs_with_pool(request)

        if body.get("stream"):
            def _chunks():
                yield from stream_image_chunks(_outputs())

            handed_off = True
            return _EditStream(_chunks, hold, _release)
        result = _finish(_outputs())
        if not encoded_images:
            raise ImageGenerationError("image is required")
        return result
    except LargeImageAdmissionError as exc:
        _clear_payload()
        raise _busy(exc) from exc
    finally:
        if not handed_off:
            _release()
