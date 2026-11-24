from __future__ import annotations

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
        bind_large_image_job,
        edit_input_guard_pixels,
        heavy_image_work,
        is_enabled,
        raise_if_request_cancelled,
    )
    from services.protocol.conversation import _admission_error_code

    guard_pixels = edit_input_guard_pixels(images, masks)

    def _prepare() -> None:
        nonlocal images, encoded_images
        if guard_pixels:
            with heavy_image_work(discovered_pixels=guard_pixels, stage="edit_input"):
                images = _composite_mask(images, masks)
                # Composite may already be inside a native call. Do not start
                # encode when the request died during that call.
                raise_if_request_cancelled()
                encoded_images = encode_images(images)
            # Encode is not interruptible. Observe cancel before any upstream.
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

    # A large or unproven input waits for one claim before decode. The raw
    # upload is already buffered by the framework; the decoded image is not
    # built while this request is only waiting. n=1 keeps that same claim
    # through generation. n>1 releases it before the per-image claims so the
    # parent slot cannot deadlock the children.
    if guard_pixels and is_enabled():
        try:
            if n == 1 and not body.get("stream"):
                with bind_large_image_job("4K"):
                    _prepare()
                    if not encoded_images:
                        raise ImageGenerationError("image is required")
                    return _finish(stream_image_outputs_with_pool(_request()))
            with bind_large_image_job("4K"):
                _prepare()
        except LargeImageAdmissionError as exc:
            raise ImageGenerationError(
                "Free-pool large-image capacity is busy. Retry this same request later; "
                "do not start a duplicate generation.",
                status_code=429,
                error_type="rate_limit_error",
                code=_admission_error_code(exc.reason),
            ) from exc
    else:
        _prepare()
    if not encoded_images:
        raise ImageGenerationError("image is required")
    return _finish(stream_image_outputs_with_pool(_request()))
