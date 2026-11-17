"""Stage 2 prompt rendering, image validation, and metadata persistence."""
from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from src.contracts.schema import CharacterResult, MainProfile, ScanResult, StageError, ensure_under
from src.utils.files import atomic_write_bytes, atomic_write_json, read_json
from src.utils.retry import is_transient_provider_error, retry_provider_call


_FIELD = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
LOGGER = logging.getLogger(__name__)


def render_prompt(template: str, values: dict[str, Any]) -> str:
    required = set(_FIELD.findall(template))
    if required != set(values):
        missing, extra = sorted(required - set(values)), sorted(set(values) - required)
        raise ValueError(f"Prompt variables mismatch (missing={missing}, extra={extra})")
    rendered = _FIELD.sub(lambda match: str(values[match.group(1)]), template)
    if re.search(r"\{[^{}]+\}", rendered):
        raise ValueError("Prompt contains an unresolved template variable")
    return rendered


def _cache_fingerprint(prompt_sha256: str, settings: dict[str, Any]) -> str:
    cache_inputs = {
        "prompt_sha256": prompt_sha256,
        "backend": settings["backend"],
        "model_id": settings["model_id"],
        "aspect_ratio": settings.get("aspect_ratio", "1:1"),
        "image_size": settings.get("image_size", "1K"),
        "output_format": settings.get("output_format", "jpg"),
        "safety_filter_level": settings.get(
            "safety_filter_level", "block_medium_and_above"
        ),
    }
    return hashlib.sha256(
        json.dumps(cache_inputs, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def seed_character_cache(
    scan: ScanResult,
    profile: MainProfile,
    settings: dict[str, Any],
    cache_dir: Path,
    image_path: Path,
    metadata_path: Path,
) -> None:
    """Promote a validated per-photo checkpoint into the shared species/color cache."""
    template = Path(settings["prompt_path"]).read_text(encoding="utf-8")
    values = {
        "species": scan.species,
        "color": scan.color,
        "visual_detail": profile.visual_detail,
        "pose": settings.get(
            "default_pose", "standing in a relaxed, welcoming pose with a gentle expression"
        ),
        "aspect_ratio": settings.get("aspect_ratio", "1:1"),
        "image_size": settings.get("image_size", "1K"),
    }
    prompt_sha256 = hashlib.sha256(render_prompt(template, values).encode("utf-8")).hexdigest()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_stem = f"{scan.species}_{scan.color}"
    cache_image = ensure_under(cache_dir / f"{cache_stem}.png", cache_dir)
    cache_metadata = ensure_under(cache_dir / f"{cache_stem}.json", cache_dir)
    character = read_json(metadata_path)
    with Image.open(image_path) as image:
        image.verify()
    with Image.open(image_path) as image:
        width, height = image.size
        if (
            image.format != "PNG"
            or min(width, height) < 256
            or max(width, height) > int(settings.get("max_dimension", 4096))
        ):
            return
    atomic_write_bytes(cache_image, image_path.read_bytes())
    atomic_write_json(
        cache_metadata,
        {
            "cache_fingerprint": _cache_fingerprint(prompt_sha256, settings),
            "model": character["model"],
            "backend": character["backend"],
            "parameters": character["parameters"],
            "request_id": character.get("request_id"),
            "width": width,
            "height": height,
        },
    )
    LOGGER.info("Seeded Stage 2 image cache: %s", cache_image.name)


def generate_character(
    scan: ScanResult, profile: MainProfile, settings: dict[str, Any], output_dir: Path, client: Any,
    cache_dir: Path | None = None,
) -> CharacterResult:
    if scan.status == "failed" or scan.species == "unknown" or scan.species not in settings["allowed_species"]:
        raise StageError("B", "unsupported_species", False, f"Cannot create a character for species {scan.species!r}")
    if scan.color is None or scan.color not in settings["allowed_colors"]:
        raise StageError("B", "missing_color", False, "Stage 1 did not produce a configured color")
    if profile.species != scan.species:
        raise StageError("B", "profile_mismatch", False, "Character profile does not match the Stage 1 species")
    template_path = Path(settings["prompt_path"])
    template = template_path.read_text(encoding="utf-8")
    values = {
        "species": scan.species, "color": scan.color, "visual_detail": profile.visual_detail,
        "pose": settings.get("default_pose", "standing in a relaxed, welcoming pose with a gentle expression"),
        "aspect_ratio": settings.get("aspect_ratio", "1:1"), "image_size": settings.get("image_size", "1K"),
    }
    prompt = render_prompt(template, values)
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_image: Path | None = None
    cache_metadata: Path | None = None
    cache_fingerprint: str | None = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_stem = f"{scan.species}_{scan.color}"
        cache_image = ensure_under(cache_dir / f"{cache_stem}.png", cache_dir)
        cache_metadata = ensure_under(cache_dir / f"{cache_stem}.json", cache_dir)
        cache_fingerprint = _cache_fingerprint(digest, settings)
        try:
            cached = read_json(cache_metadata)
            if cached.get("cache_fingerprint") == cache_fingerprint:
                with Image.open(cache_image) as image:
                    image.verify()
                with Image.open(cache_image) as image:
                    width, height = image.size
                    usable = (
                        image.format == "PNG"
                        and min(width, height) >= 256
                        and max(width, height) <= int(settings.get("max_dimension", 4096))
                    )
                if usable:
                    data = cache_image.read_bytes()
                    image_path = ensure_under(output_dir / "character.png", output_dir)
                    atomic_write_bytes(image_path, data)
                    result = CharacterResult(
                        schema_version="1.0", image_id=scan.image_id,
                        species=scan.species, color=scan.color,
                        profile_id=profile.species, name=profile.name,
                        image_path=image_path.name, model=cached["model"],
                        backend=cached["backend"], parameters=cached["parameters"],
                        prompt_sha256=digest, request_id=cached.get("request_id"),
                        width=width, height=height, mime_type="image/png",
                    )
                    atomic_write_json(output_dir / "character.json", result.to_dict())
                    LOGGER.info("Reused Stage 2 image cache: %s", cache_image.name)
                    return result
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            pass

    provider_settings = {
        **settings,
        "image_id": scan.image_id,
        "species": scan.species,
        "color": scan.color,
    }
    retry_log = settings.get("on_retry")
    try:
        response = retry_provider_call(
            lambda: client.generate(prompt, output_dir=output_dir, settings=provider_settings),
            max_attempts=int(settings.get("max_attempts", 3)),
            base_delay_seconds=float(settings.get("backoff_seconds", 1)),
            on_retry=retry_log,
        )
    except StageError:
        raise
    except Exception as error:
        status = getattr(error, "status_code", None)
        raise StageError("B", "provider_error", is_transient_provider_error(error), f"{type(error).__name__}: {error}") from error
    data = response.data
    if not isinstance(data, bytes) or not data or len(data) > int(settings.get("max_image_bytes", 20_000_000)):
        raise StageError("B", "invalid_image", False, "Image response is empty or exceeds max_image_bytes")
    if response.mime_type not in {"image/png", "image/jpeg", "image/webp"}:
        raise StageError("B", "invalid_mime_type", False, f"Unsupported image MIME type: {response.mime_type}")
    image_path = ensure_under(output_dir / "character.png", output_dir)
    try:
        from io import BytesIO
        with Image.open(BytesIO(data)) as image:
            image.verify()
        with Image.open(BytesIO(data)) as image:
            width, height = image.size
            if min(width, height) < 256 or max(width, height) > int(settings.get("max_dimension", 4096)):
                raise StageError("B", "invalid_dimensions", False, f"Unusable image dimensions: {width}x{height}")
            if image.format not in {"PNG", "JPEG", "WEBP"}:
                raise StageError("B", "invalid_image", False, f"Unexpected decoded format: {image.format}")
            png = BytesIO()
            image.convert("RGBA").save(png, format="PNG", optimize=True)
            data = png.getvalue()
    except (OSError, UnidentifiedImageError) as error:
        raise StageError("B", "image_decode", False, "Provider returned invalid image bytes") from error
    atomic_write_bytes(image_path, data)
    parameters = {
        "aspect_ratio": settings.get("aspect_ratio", "1:1"),
        "image_size": settings.get("image_size", "1K"),
        "seed": response.seed,
        "seed_supported": response.seed_supported,
    }
    result = CharacterResult(
        schema_version="1.0", image_id=scan.image_id, species=scan.species, color=scan.color,
        profile_id=profile.species, name=profile.name, image_path=image_path.name,
        model=response.model, backend=str(settings["backend"]),
        parameters=parameters,
        prompt_sha256=digest, request_id=response.request_id, width=width, height=height,
        mime_type="image/png",
    )
    atomic_write_json(output_dir / "character.json", result.to_dict())
    if cache_image is not None and cache_metadata is not None and cache_fingerprint is not None:
        atomic_write_bytes(cache_image, data)
        atomic_write_json(
            cache_metadata,
            {
                "cache_fingerprint": cache_fingerprint,
                "model": response.model,
                "backend": str(settings["backend"]),
                "parameters": parameters,
                "request_id": response.request_id,
                "width": width,
                "height": height,
            },
        )
        LOGGER.info("Saved Stage 2 image cache: %s", cache_image.name)
    return result
