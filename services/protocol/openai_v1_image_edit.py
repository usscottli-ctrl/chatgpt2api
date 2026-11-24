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
        buf = BytesIO()
        img.save(buf, format="PNG")
        result.append((buf.getvalue(), filename, "image/png"))
    return result


def handle(body: dict[str, Any]) -> dict[str, Any] | Iterator[dict[str, Any]]:
    prompt = str(body.get("prompt") or "")
    images = body.get("images") or []
    masks = body.get("mask") or []
    from services.free_large_image_guard import (
        LargeImageAdmissionError,
        base64_payload_length,
        bind_edit_prepare,
        bind_large_image_job,
        composite_output_upper_bound,
        edit_input_guard_pixels,
        heavy_image_work,
        is_enabled,
        payload_byte_length,
        raise_if_request_cancelled,
        reserve_edit_buffers,
        reset_edit_prepare,
    )
    from services.protocol.conversation import _admission_error_code

    guard_pixels = edit_input_guard_pixels(images, masks)
    hold = None

    def _prepare() -> None:
        nonlocal images, encoded_images
        if guard_pixels:
            with heavy_image_work(discovered_pixels=guard_pixels, stage="edit_input"):
                if masks and hold is not None:
                    upper = composite_output_upper_bound(images)
                    hold.add(upper)
                    produced = _composite_mask(images, masks)
                    actual = payload_byte_length(produced)
                    if actual > upper:
                        produced = None
                        raise LargeImageAdmissionError("buffer_budget")
                    hold.release_amount(upper - actual)
                    previous = images
                    images = produced
                    previous = None
                    del previous
                else:
                    images = _composite_mask(images, masks)
                raise_if_request_cancelled()
                if hold is not None:
                    encoded_need = 0
                    for item in images or []:
                        raw = item[0] if isinstance(item, tuple) and item else b""
                        if raw:
                            encoded_need += base64_payload_length(len(raw))
                    hold.add(encoded_need)
                encoded_images = encode_images(images)
            raise_if_request_cancelled()
        else:
            images = _composite_mask(images, masks)
            encoded_images = encode_images(images)

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
    def _clear_payload() -> None:
        nonlocal images, encoded_images
        images = []
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

    try:
        if n == 1 and not body.get("stream"):
            with bind_large_image_job("4K"):
                _prepare()
                if not encoded_images:
                    raise ImageGenerationError("image is required")
                return _finish(stream_image_outputs_with_pool(_request()))

        gate = threading.Lock()
        state: dict[str, Any] = {"done": False, "error": None}

        def _prepare_once() -> None:
            with gate:
                if state["error"] is not None:
                    raise state["error"]
                if state["done"]:
                    return
                try:
                    _prepare()
                    state["done"] = True
                except BaseException as exc:
                    state["error"] = exc
                    raise

        if body.get("stream"):
            def _stream():
                prepare_token = bind_edit_prepare(_prepare_once)
                try:
                    yield from stream_image_chunks(stream_image_outputs_with_pool(_request()))
                finally:
                    reset_edit_prepare(prepare_token)
                    _release()

            return _stream()
        prepare_token = bind_edit_prepare(_prepare_once)
        try:
            result = _finish(stream_image_outputs_with_pool(_request()))
        finally:
            reset_edit_prepare(prepare_token)
        if not encoded_images:
            raise ImageGenerationError("image is required")
        return result
    except LargeImageAdmissionError as exc:
        raise _busy(exc) from exc
    finally:
        if not body.get("stream"):
            _release()
