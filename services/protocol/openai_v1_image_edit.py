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
        if self.tell() + len(data) > self._limit:
            self.seek(0)
            self.truncate(0)
            from services.free_large_image_guard import LargeImageAdmissionError

            raise LargeImageAdmissionError("buffer_budget")
        return super().write(data)


class _EditStream:
    """One edit stream and the buffer lease it owns.

    close() is idempotent from any thread. It marks the lease cancelled and
    releases immediately only when no native step is inside next(). A running
    encode keeps the bytes until it returns, and that return does not start
    upstream. Garbage collection uses the same close.
    """

    def __init__(self, factory, hold, cleanup) -> None:
        self._factory = factory
        self._hold = hold
        self._cleanup = cleanup
        self._gen = None
        self._closed = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._closed and self._gen is None:
            raise StopIteration
        self._hold.enter_native()
        try:
            if self._gen is None:
                self._gen = self._factory()
            return next(self._gen)
        except StopIteration:
            self._closed = True
            raise
        finally:
            self._hold.leave_native()

    def close(self) -> None:
        self._closed = True
        self._hold.request_close(self._cleanup)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _composite_mask(
    images: list[tuple[bytes, str, str]],
    masks: list[tuple[bytes, str, str]],
) -> list[tuple[bytes, str, str]]:
    """将 mask 的 alpha 通道合成到图片中，标识需要编辑的区域。
    
    mask 的透明区域（低 alpha）= 需要编辑的区域，
    mask 的不透明区域（高 alpha）= 保留的区域。
    如果无 mask 则返回原图。
    """
    if not masks:
        return images
    result: list[tuple[bytes, str, str]] = []
    for i, (data, filename, mime_type) in enumerate(images):
        mask_data = masks[i][0] if i < len(masks) else masks[-1][0]
        img = Image.open(BytesIO(data)).convert("RGBA")
        mask_img = Image.open(BytesIO(mask_data))
        if mask_img.mode == "RGBA":
            alpha = mask_img.split()[3]
        elif mask_img.mode == "L":
            alpha = mask_img
        else:
            alpha = mask_img.convert("L")
        alpha = alpha.resize(img.size, Image.LANCZOS)
        img.putalpha(alpha)
        from services.free_large_image_guard import icc_chunk_upper_bound, png_rgba_upper_bound

        icc = img.info.get("icc_profile") if isinstance(getattr(img, "info", None), dict) else None
        extra = icc_chunk_upper_bound(icc) if isinstance(icc, (bytes, bytearray)) else 0
        limit = png_rgba_upper_bound(img.size[0], img.size[1], extra)
        buf = _CappedBytesIO(limit)
        if isinstance(icc, (bytes, bytearray)):
            img.save(buf, format="PNG", icc_profile=icc)
        else:
            img.save(buf, format="PNG")
        result.append((buf.getvalue(), filename, "image/png"))
        for opened in (img, mask_img):
            closer = getattr(opened, "close", None)
            if closer is not None:
                closer()
    return result


def handle(body: dict[str, Any]) -> dict[str, Any] | Iterator[dict[str, Any]]:
    prompt = str(body.get("prompt") or "")
    images = body.get("images") or []
    masks = body.get("mask") or []
    from services.free_large_image_guard import (
        LargeImageAdmissionError,
        base64_payload_length,
        bind_large_image_job,
        composite_output_upper_bound,
        edit_input_guard_pixels,
        heavy_image_work,
        is_enabled,
        payload_byte_length,
        raise_if_request_cancelled,
        png_rgba_upper_bound,
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
        try:
            if guard_pixels:
                with heavy_image_work(discovered_pixels=guard_pixels, stage="edit_input"):
                    if masks and hold is not None:
                        upper = composite_output_upper_bound(images)
                        if upper is None:
                            discovered = []
                            for item in images:
                                part = item[0] if isinstance(item, tuple) and item else b""
                                with Image.open(BytesIO(part)) as opened:
                                    opened.load()
                                    discovered.append(opened.size)
                            item = None
                            part = None
                            upper = sum(png_rgba_upper_bound(w, h) for w, h in discovered)
                        hold.add(upper)
                        produced = _composite_mask(images, masks)
                        actual = payload_byte_length(produced)
                        if actual > upper:
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
                    encoded_images = encode_images(images)
                raise_if_request_cancelled()
            else:
                images = _composite_mask(images, masks)
                encoded_images = encode_images(images)
        finally:
            if isinstance(produced, list):
                produced.clear()
            produced = None
            item = None
            part = None

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
