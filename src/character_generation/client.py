"""Replicate image generation adapter."""
from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from src.contracts.schema import StageError
from src.utils.retry import is_transient_provider_error, retry_call

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImageResponse:
    data: bytes
    mime_type: str
    model: str
    request_id: str | None = None
    seed: int | None = None
    seed_supported: bool = False


class ImageClient(Protocol):
    def generate(self, prompt: str, *, output_dir: Path, settings: dict[str, Any]) -> ImageResponse: ...


def _provider_error(error: Exception, stage: str) -> StageError:
    status = getattr(error, "status_code", None)
    if status is None:
        status = getattr(error, "status", None)
    if status is None:
        status = getattr(getattr(error, "response", None), "status_code", None)
    retryable = status in (408, 409, 425, 429) or (isinstance(status, int) and status >= 500)
    name = type(error).__name__.lower()
    retryable = retryable or any(
        token in name for token in ("timeout", "connection", "connect", "ssl")
    )
    if status in (401, 403):
        category = "credentials"
        retryable = False
    elif status == 429:
        category = "rate_limit"
    elif status == 400:
        category = "invalid_request"
        retryable = False
    elif retryable:
        category = "provider_transient"
    else:
        category = "provider_error"
    return StageError(stage, category, retryable, f"{type(error).__name__}: {error}")


class ReplicateImageClient:
    def __init__(
        self, model_id: str, timeout_seconds: float = 120,
        api_token: str | None = None,
    ) -> None:
        self.api_token = api_token or os.environ.get("REPLICATE_API_TOKEN")
        if not self.api_token:
            raise StageError("B", "missing_credentials", False, "REPLICATE_API_TOKEN is required for image generation")
        try:
            from replicate import Client
        except ImportError as error:
            raise StageError("B", "dependency_missing", False, "Install the replicate dependency") from error
        self.client = Client(api_token=self.api_token, timeout=timeout_seconds)
        self.model_id = model_id
        self.timeout_seconds = timeout_seconds

    def generate(self, prompt: str, *, output_dir: Path, settings: dict[str, Any]) -> ImageResponse:
        del output_dir  # The image is returned in memory and persisted by the caller.
        if settings.get("reference_image_path") or settings.get("reference_image_paths"):
            raise StageError(
                "B", "unsupported_input", False,
                "The configured Replicate model does not accept reference image inputs",
            )

        model_input = {
            "prompt": prompt,
            "aspect_ratio": str(settings.get("aspect_ratio", "1:1")),
            "image_size": str(settings.get("image_size", "1K")),
            "output_format": str(settings.get("output_format", "jpg")),
            "safety_filter_level": str(settings.get("safety_filter_level", "block_medium_and_above")),
        }
        wait_seconds = max(1, min(60, int(self.timeout_seconds)))
        image_id = settings.get("image_id", "unknown")
        try:
            LOGGER.info(
                "Photo %s | Stage 2 | submitting %s prediction to Replicate",
                image_id,
                self.model_id,
            )
            output = self.client.run(self.model_id, input=model_input, wait=wait_seconds)
            read = getattr(output, "read", None)
            if not callable(read):
                raise StageError("B", "empty_image", False, "Replicate returned no readable image file")
            LOGGER.info(
                "Photo %s | Stage 2 | prediction finished; downloading generated image",
                image_id,
            )
            download_attempts = max(1, int(settings.get("max_attempts", 3)))
            download_delay = max(0, float(settings.get("backoff_seconds", 1)))

            def log_download_retry(attempt: int, error: Exception) -> None:
                LOGGER.warning(
                    "Photo %s | Stage 2 | image download retry %d/%d: %s",
                    image_id,
                    attempt + 1,
                    download_attempts,
                    error,
                )

            try:
                image_data = retry_call(
                    read,
                    max_attempts=download_attempts,
                    delay_seconds=download_delay,
                    should_retry=is_transient_provider_error,
                    on_retry=log_download_retry,
                )
            except Exception as error:
                # The prediction already completed. Do not submit another paid
                # prediction just because fetching its output had a transient error.
                raise StageError(
                    "B",
                    "image_download_failed",
                    False,
                    f"Replicate prediction completed, but its image could not be downloaded after {download_attempts} attempts: {type(error).__name__}: {error}",
                ) from error
            if not isinstance(image_data, bytes) or not image_data:
                raise StageError("B", "empty_image", False, "Replicate returned empty image data")
            LOGGER.info(
                "Photo %s | Stage 2 | downloaded generated image (%d bytes)",
                image_id,
                len(image_data),
            )
            mime_type = getattr(output, "content_type", None)
            if not isinstance(mime_type, str) or not mime_type:
                mime_type = {
                    "jpg": "image/jpeg",
                    "jpeg": "image/jpeg",
                    "png": "image/png",
                    "webp": "image/webp",
                }.get(model_input["output_format"].lower(), "application/octet-stream")
            return ImageResponse(
                data=image_data,
                mime_type=mime_type,
                model=self.model_id,
            )
        except StageError:
            raise
        except Exception as error:
            LOGGER.exception(
                "Photo %s | Stage 2 | Replicate prediction or image download failed",
                image_id,
            )
            raise _provider_error(error, "B") from error


def create_image_client(settings: dict[str, Any]) -> ImageClient:
    backend = settings.get("backend")
    if backend != "replicate":
        raise StageError("B", "invalid_backend", False, f"Unsupported image backend: {backend}")
    return ReplicateImageClient(
        str(settings["model_id"]),
        float(settings.get("timeout_seconds", 120)),
    )
