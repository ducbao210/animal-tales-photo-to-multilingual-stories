from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml
from PIL import Image, UnidentifiedImageError

from src.animal_scan.config import load_config
from src.animal_scan.data import discover_images
from src.animal_scan.pipeline import PartAPipeline
from src.character_generation.client import create_image_client
from src.character_generation.generator import generate_character, seed_character_cache
from src.contracts.schema import ContractError, StageError, StoryResult, validate_scan
from src.display_ui.gallery import launch_streamlit_gallery
from src.story_generation.client import GeminiStoryClient
from src.story_generation.generator import generate_story
from src.translation.client import GroqTranslationClient
from src.translation.translator import back_translate_story, translate_story
from src.utils.configuration import load_pipeline_config
from src.utils.files import atomic_write_json, read_json
from src.utils.logger import EventLogger, PipelineEventLogger, configure_logging
from src.utils.state import RunState, sha256_file, stable_fingerprint

PROJECT_ROOT = Path(__file__).resolve().parent
LOGGER = logging.getLogger("[MVP] - Let's story")


def _log_step(
    events: EventLogger,
    *,
    run_id: str,
    image_id: str,
    step: str,
    operation: Callable[[], Any],
    locale: str | None = None,
) -> Any:
    started = time.perf_counter()
    common = {"run_id": run_id, "image_id": image_id, "step": step}
    if locale:
        common["locale"] = locale
    events.write({**common, "event": "step_start", "attempt": 1, "status": "running"})
    try:
        result = operation()
    except StageError as error:
        events.write(
            {
                **common,
                "event": "step_end",
                "status": "failed",
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                **error.to_dict(),
            }
        )
        raise
    except Exception as error:
        events.write(
            {
                **common,
                "event": "step_end",
                "status": "failed",
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                "category": "unexpected",
                "error_type": type(error).__name__,
                "message": str(error),
            }
        )
        raise
    events.write(
        {
            **common,
            "event": "step_end",
            "status": "succeeded",
            "attempt": 1,
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
        }
    )
    return result


def _read_scan(path: Path, species: set[str], colors: set[str], unknown: str) -> Any:
    return validate_scan(read_json(path), species, colors, unknown)


def _valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, UnidentifiedImageError):
        return False


def _valid_json(path: Path) -> bool:
    try:
        read_json(path)
        return True
    except (OSError, ValueError):
        return False


def _valid_context_artifact(path: Path) -> bool:
    if path.suffix.lower() == ".png":
        return _valid_image(path)
    return _valid_json(path)


def _valid_back_translation(path: Path) -> bool:
    try:
        data = read_json(path)
        pages = data.get("pages")
        return (
            isinstance(pages, list)
            and [page.get("page") for page in pages] == list(range(1, 9))
            and isinstance(data.get("differences"), list)
        )
    except (OSError, ValueError, AttributeError, TypeError):
        return False


def _valid_character(root: Path, scan: Any) -> bool:
    try:
        data = read_json(root / "character.json")
        image = root / "character.png"
        if not _valid_image(image):
            return False
        return (
            data.get("schema_version") == "1.0"
            and data.get("image_id") == scan.image_id
            and data.get("species") == scan.species
            and data.get("color") == scan.color
            and data.get("status") == "complete"
        )
    except (OSError, ValueError, AttributeError):
        return False


def _valid_story(root: Path, scan: Any) -> bool:
    try:
        story = StoryResult.from_dict(read_json(root / "story_en.json"))
        return (
            story.image_id == scan.image_id
            and story.species == scan.species
            and story.color == scan.color
            and len(story.pages) == 8
            and story.validation.get("passed") is True
            and [page.page for page in story.pages] == list(range(1, 9))
            and [page.section for page in story.pages]
            == [
                section
                for section in ("beginning", "problem", "solution", "happy_ending")
                for _ in range(2)
            ]
        )
    except (OSError, ValueError, ContractError, AttributeError):
        return False


def _valid_translations(root: Path) -> bool:
    try:
        bundle = read_json(root / "translations.json")
        translations = bundle.get("translations", {})
        if bundle.get("schema_version") != "1.0" or set(translations) != {
            "fr-FR",
            "es-ES",
            "vi-VN",
        }:
            return False
        return bundle.get("validation", {}).get("passed") is True and all(
            len(translations[locale].get("pages", [])) == 8
            and [page.get("page") for page in translations[locale]["pages"]]
            == list(range(1, 9))
            and translations[locale].get("validation", {}).get("passed") is True
            for locale in translations
        )
    except (OSError, ValueError, AttributeError, TypeError):
        return False


def _stage_error_status(error: StageError) -> str:
    """Keep a saved-but-incomplete translation run distinct from total failure."""
    if error.step == "D" and error.category == "partial_translation":
        return "partial"
    return "failed"


def _prompt_digest(paths: list[str]) -> list[str]:
    result = []
    for path in paths:
        result.append(hashlib.sha256(Path(path).read_bytes()).hexdigest())
    return result


def run_pipeline(
    config_path: Path,
    input_dir: Path | None = None,
    limit: int | None = None,
    refresh_speciesnet: bool = False,
    resume_override: bool | None = None,
    support_species: str | None = None,
    support_color: str | None = None,
    areas: list[str] | None = None,
    generate_context_images: bool = False,
    back_translation: bool = False,
) -> int:
    root, document, world = load_pipeline_config(config_path)
    configured_areas = set(world.areas_by_id())
    if areas is not None and (len(areas) < 2 or len(set(areas)) != len(areas) or not set(areas) <= configured_areas):
        raise ContractError("--areas must contain at least two distinct configured area IDs")
    config = load_config(config_path)
    part_e = document["part_e"]
    if resume_override is not None:
        part_e["resume"] = resume_override
    input_dir = input_dir or config.paths["input_dir"]
    if not input_dir.is_dir():
        raise ContractError(f"Input directory does not exist: {input_dir}")
    image_paths = list(
        discover_images(input_dir, tuple(config.values["image_extensions"]), limit)
    )
    if not image_paths:
        raise ContractError(f"No supported images found in {input_dir}")

    # Instantiate configured clients before the local vision models so missing credentials fail early.
    image_settings = dict(document["part_b"])
    image_settings.update(
        {
            "allowed_species": set(config.taxonomy["species"]),
            "allowed_colors": set(config.taxonomy["colors"]),
            "prompt_path": str(root / document["part_b"]["prompt_path"]),
        }
    )
    image_client = create_image_client(image_settings)
    story_settings = dict(document["part_c"])
    story_settings.update({
        "allowed_colors": set(config.taxonomy["colors"]),
        "support_species": support_species,
        "support_color": support_color,
        "selected_areas": areas,
    })
    translation_settings = dict(document["part_d"])
    translation_settings["age"] = story_settings.get("age", "5-7")
    story_client = GeminiStoryClient(
        str(story_settings["model_id"]), float(story_settings["timeout_seconds"])
    )
    translation_client = GroqTranslationClient(
        str(translation_settings["model_id"]),
        float(translation_settings["timeout_seconds"]),
        reasoning_effort=translation_settings.get("reasoning_effort"),
        requests_per_minute=int(translation_settings["groq_rate_limits"]["requests_per_minute"]),
        tokens_per_minute=int(translation_settings["groq_rate_limits"]["tokens_per_minute"]),
    )
    for settings in (story_settings, translation_settings):
        for key in list(settings):
            if key.startswith("prompt_"):
                settings[key] = str(root / settings[key])

    output_root = Path(part_e["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    log_root = Path(part_e["log_dir"])
    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + uuid.uuid4().hex[:6]
    )
    events = PipelineEventLogger(log_root, run_id)
    events.write(
        {
            "run_id": run_id,
            "event": "run_start",
            "photo_count": len(image_paths),
            "input_dir": str(input_dir),
        }
    )
    try:
        part_a = PartAPipeline(config, refresh_speciesnet=refresh_speciesnet)
    except Exception as error:
        events.write(
            {
                "run_id": run_id,
                "event": "run_end",
                "status": "setup_failed",
                "error_type": type(error).__name__,
                "message": str(error),
            }
        )
        LOGGER.error("Stage 1 setup failed: %s", error)
        return 2
    species = set(config.taxonomy["species"])
    colors = set(config.taxonomy["colors"])
    unknown = config.taxonomy["unknown_species"]
    summaries: list[dict[str, Any]] = []

    for image_path in image_paths:
        image_id = part_a._image_id(image_path)
        image_sha = sha256_file(image_path)
        photo_root = output_root / image_id
        photo_root.mkdir(parents=True, exist_ok=True)
        manifest_path = photo_root / "manifest.json"
        state = RunState(manifest_path, image_id, image_sha)
        context_image_failure: StageError | None = None
        try:
            part_a_fp = stable_fingerprint(image_sha, document["part_a"])
            scan_path = photo_root / "scan.json"
            cached_a = bool(part_e.get("resume", True)) and state.is_complete(
                "A", part_a_fp, photo_root, _valid_json
            )
            if cached_a:
                scan = _read_scan(scan_path, species, colors, unknown)
                events.write(
                    {
                        "run_id": run_id,
                        "image_id": image_id,
                        "step": "A",
                        "event": "step_start",
                        "status": "running",
                    }
                )
                events.write(
                    {
                        "run_id": run_id,
                        "image_id": image_id,
                        "step": "A",
                        "event": "step_end",
                        "status": "skipped",
                        "duration_ms": 0,
                        "reason": "validated checkpoint",
                    }
                )
            else:
                raw_scan = _log_step(
                    events,
                    run_id=run_id,
                    image_id=image_id,
                    step="A",
                    operation=lambda: part_a.process_one(image_path),
                )
                scan = validate_scan(raw_scan, species, colors, unknown)
                if scan.status == "partial":
                    partial_reasons = (
                        raw_scan.get("mapping_notes")
                        or raw_scan.get("processing_notes")
                        or raw_scan.get("errors")
                        or ["Stage 1 returned a partial result."]
                    )
                    LOGGER.warning(
                        "Stage 1 partial result for %s (%s): %s",
                        image_id,
                        raw_scan.get("input_path", "input path unavailable"),
                        "; ".join(str(reason) for reason in partial_reasons),
                    )
                atomic_write_json(scan_path, scan.to_dict())
                state.complete("A", part_a_fp, [scan_path], photo_root)

            profile = world.main.get(scan.species)
            b_prompt = image_settings["prompt_path"]
            b_fp = stable_fingerprint(
                image_sha,
                document["part_b"],
                _prompt_digest([b_prompt]),
                scan.to_dict(),
                profile.__dict__ if profile else None,
            )
            b_artifacts = [photo_root / "character.png", photo_root / "character.json"]
            cached_b = (
                bool(part_e.get("resume", True))
                and state.is_complete("B", b_fp, photo_root)
                and _valid_character(photo_root, scan)
            )
            if cached_b:
                char_data = read_json(photo_root / "character.json")
                from src.contracts.schema import CharacterResult

                character = CharacterResult(**char_data)
                if profile is not None:
                    try:
                        seed_character_cache(
                            scan,
                            profile,
                            image_settings,
                            output_root / "character_cache",
                            photo_root / "character.png",
                            photo_root / "character.json",
                        )
                    except (OSError, ValueError, KeyError, TypeError) as error:
                        LOGGER.warning(
                            "Could not seed Stage 2 image cache for %s: %s",
                            image_id,
                            error,
                        )
                events.write(
                    {
                        "run_id": run_id,
                        "image_id": image_id,
                        "step": "B",
                        "event": "step_end",
                        "status": "skipped",
                        "reason": "validated checkpoint",
                    }
                )
            else:
                if profile is None:
                    raise StageError(
                        "B",
                        "unsupported_species",
                        False,
                        f"No character profile for Stage 1 species {scan.species!r}",
                    )
                image_settings["on_retry"] = lambda attempt, error: events.write(
                    {
                        "run_id": run_id,
                        "image_id": image_id,
                        "step": "B",
                        "event": "retry",
                        "attempt": attempt + 1,
                        "error_type": type(error).__name__,
                        "message": str(error),
                    }
                )
                character = _log_step(
                    events,
                    run_id=run_id,
                    image_id=image_id,
                    step="B",
                    operation=lambda: generate_character(
                        scan,
                        profile,
                        image_settings,
                        photo_root,
                        image_client,
                        cache_dir=output_root / "character_cache",
                    ),
                )
                image_settings.pop("on_retry", None)
                state.complete("B", b_fp, b_artifacts, photo_root)

            c_paths = [
                story_settings[k]
                for k in (
                    "prompt_outline",
                    "prompt_write",
                    "prompt_review",
                    "prompt_refine",
                    "prompt_page_illustration",
                )
            ]
            part_c_fingerprint_settings = {
                key: value
                for key, value in document["part_c"].items()
                if key not in {"prompt_phase_plan", "prompt_context"}
            }
            c_fp = stable_fingerprint(
                image_sha,
                part_c_fingerprint_settings,
                _prompt_digest(c_paths),
                Path(part_e["world_path"]).read_bytes(),
                support_species,
                support_color,
                areas,
                scan.to_dict(),
                sha256_file(photo_root / "character.png"),
            )
            c_artifacts = [
                photo_root / "story_en.json",
                photo_root / "page_illustration_prompts.json",
            ]
            cached_c = (
                bool(part_e.get("resume", True))
                and state.is_complete("C", c_fp, photo_root)
                and _valid_story(photo_root, scan)
            )
            if cached_c:
                story = StoryResult.from_dict(read_json(photo_root / "story_en.json"))
                events.write(
                    {
                        "run_id": run_id,
                        "image_id": image_id,
                        "step": "C",
                        "event": "step_end",
                        "status": "skipped",
                        "reason": "validated checkpoint",
                    }
                )
            else:
                story_settings["on_retry"] = lambda attempt, error: events.write(
                    {
                        "run_id": run_id,
                        "image_id": image_id,
                        "step": "C",
                        "event": "retry",
                        "attempt": attempt + 1,
                        "error_type": type(error).__name__,
                        "message": str(error),
                    }
                )
                story = _log_step(
                    events,
                    run_id=run_id,
                    image_id=image_id,
                    step="C",
                    operation=lambda: generate_story(
                        scan,
                        character,
                        world,
                        story_settings,
                        photo_root,
                        story_client,
                    ),
                )
                state.complete("C", c_fp, c_artifacts, photo_root)

            if generate_context_images:
                from src.character_generation.context_images import generate_context_images as generate_phase_images

                context_image_artifacts = [
                    photo_root / "phase_visual_plan.json",
                    photo_root / "phase_image_prompts.json",
                    *(
                        photo_root / f"phase_{section}.png"
                        for section in ("beginning", "problem", "solution", "happy_ending")
                    ),
                ]
                context_image_fp = stable_fingerprint(
                    c_fp,
                    sha256_file(photo_root / "story_en.json"),
                    document["part_c"]["prompt_phase_plan"],
                    document["part_c"]["prompt_context"],
                    _prompt_digest(
                        [story_settings["prompt_phase_plan"], story_settings["prompt_context"]]
                    ),
                    document["part_b"],
                )
                if (
                    bool(part_e.get("resume", True))
                    and state.is_complete(
                        "C_images", context_image_fp, photo_root, _valid_context_artifact
                    )
                ):
                    events.write({
                        "run_id": run_id,
                        "image_id": image_id,
                        "step": "C_images",
                        "event": "step_end",
                        "status": "skipped",
                        "reason": "validated checkpoint",
                    })
                else:
                    try:
                        _log_step(
                            events,
                            run_id=run_id,
                            image_id=image_id,
                            step="C_images",
                            operation=lambda: generate_phase_images(
                                story, world, story_settings, image_settings, photo_root,
                                story_client, image_client,
                            ),
                        )
                    except StageError as error:
                        context_image_failure = StageError(
                            "C_images", error.category, error.retryable, error.message
                        )
                        LOGGER.warning(
                            "Photo %s | C_images | %s; continuing to Stage 4",
                            image_id, error.message,
                        )
                    else:
                        state.complete(
                            "C_images", context_image_fp, context_image_artifacts,
                            photo_root,
                        )
            story_settings.pop("on_retry", None)

            d_paths = [
                translation_settings[k]
                for k in ("prompt_translate", "prompt_review", "prompt_refine")
            ]
            part_d_fingerprint_settings = {
                key: value
                for key, value in document["part_d"].items()
                if key not in {"back_translation", "prompt_back_translation"}
            }
            d_fp = stable_fingerprint(
                image_sha,
                part_d_fingerprint_settings,
                translation_settings.get("age", "5-7"),
                _prompt_digest(d_paths),
                Path(part_e["world_path"]).read_bytes(),
                sha256_file(photo_root / "story_en.json"),
                story_settings["model_id"],
            )
            d_artifacts = [
                photo_root / "translations.json",
                *(
                    photo_root / f"{locale}.json"
                    for locale in document["part_d"]["locales"]
                ),
            ]
            if (
                bool(part_e.get("resume", True))
                and state.is_complete("D", d_fp, photo_root)
                and _valid_translations(photo_root)
            ):
                events.write(
                    {
                        "run_id": run_id,
                        "image_id": image_id,
                        "step": "D",
                        "event": "step_end",
                        "status": "skipped",
                        "reason": "validated checkpoint",
                    }
                )
            else:
                translation_settings["on_event"] = lambda item: events.write(
                    {"run_id": run_id, "image_id": image_id, "step": "D", **item}
                )
                translation_settings["on_retry"] = lambda attempt, error: events.write(
                    {
                        "run_id": run_id,
                        "image_id": image_id,
                        "step": "D",
                        "event": "retry",
                        "attempt": attempt + 1,
                        "error_type": type(error).__name__,
                    }
                )
                _log_step(
                    events,
                    run_id=run_id,
                    image_id=image_id,
                    step="D",
                    operation=lambda: translate_story(
                        story,
                        world,
                        translation_settings,
                        photo_root,
                        translation_client,
                        reviewer_client=translation_client,
                        cache_fingerprint=d_fp,
                        reuse_cache=bool(part_e.get("resume", True)),
                    ),
                )
                translation_settings.pop("on_event", None)
                translation_settings.pop("on_retry", None)
                if not _valid_translations(photo_root):
                    raise StageError(
                        "D",
                        "invalid_bundle",
                        False,
                        "Translation outputs did not pass bundle validation",
                    )
                state.complete("D", d_fp, d_artifacts, photo_root)

            if back_translation:
                back_translation_fp = stable_fingerprint(
                    d_fp,
                    translation_settings["model_id"],
                    translation_settings.get("max_completion_tokens"),
                    translation_settings.get("temperature"),
                    _prompt_digest([translation_settings["prompt_back_translation"]]),
                    *(
                        sha256_file(photo_root / f"{locale}.json")
                        for locale in document["part_d"]["locales"]
                    ),
                )
                back_translation_artifacts = [
                    photo_root / f"back_translation_{locale}.json"
                    for locale in document["part_d"]["locales"]
                ]
                if (
                    bool(part_e.get("resume", True))
                    and state.is_complete(
                        "D_back_translation", back_translation_fp, photo_root,
                        _valid_back_translation,
                    )
                ):
                    events.write({
                        "run_id": run_id,
                        "image_id": image_id,
                        "step": "D_back_translation",
                        "event": "step_end",
                        "status": "skipped",
                        "reason": "validated checkpoint",
                    })
                else:
                    _log_step(
                        events,
                        run_id=run_id,
                        image_id=image_id,
                        step="D_back_translation",
                        operation=lambda: back_translate_story(
                            story, translation_settings, photo_root, translation_client
                        ),
                    )
                    state.complete(
                        "D_back_translation", back_translation_fp,
                        back_translation_artifacts, photo_root,
                    )
            if context_image_failure is not None:
                failure = context_image_failure.to_dict()
                failure["image_id"] = image_id
                atomic_write_json(photo_root / "failure.json", failure)
                summaries.append({
                    "image_id": image_id, "status": "partial",
                    "stage": "C_images", "category": context_image_failure.category,
                    "output_dir": str(photo_root),
                })
            else:
                summaries.append(
                    {
                        "image_id": image_id,
                        "status": "complete",
                        "output_dir": str(photo_root),
                    }
                )
        except StageError as error:
            status = _stage_error_status(error)
            failure = error.to_dict()
            failure["image_id"] = image_id
            state.failed(failure)
            atomic_write_json(photo_root / "failure.json", failure)
            if status == "partial":
                LOGGER.warning(
                    "Photo %s partially completed at %s (%s): %s",
                    image_id, error.step, error.category, error.message,
                )
            else:
                LOGGER.error(
                    "Photo %s failed at %s (%s): %s",
                    image_id, error.step, error.category, error.message,
                )
            summaries.append(
                {
                    "image_id": image_id,
                    "status": status,
                    "stage": error.step,
                    "category": error.category,
                }
            )
        except Exception as error:
            failure = {
                "image_id": image_id,
                "stage": "pipeline",
                "category": "unexpected",
                "message": str(error),
                "error_type": type(error).__name__,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            state.failed(failure)
            atomic_write_json(photo_root / "failure.json", failure)
            LOGGER.exception("Photo %s failed unexpectedly", image_id)
            summaries.append(
                {
                    "image_id": image_id,
                    "status": "failed",
                    "stage": "pipeline",
                    "category": "unexpected",
                }
            )

    events.write({"run_id": run_id, "event": "run_end", "results": summaries})
    complete = sum(item["status"] == "complete" for item in summaries)
    partial = sum(item["status"] == "partial" for item in summaries)
    failed = sum(item["status"] == "failed" for item in summaries)
    LOGGER.info(
        "Pipeline run %s: %d complete, %d partial, %d failed, %d requested.",
        run_id,
        complete,
        partial,
        failed,
        len(summaries),
    )
    return 1 if partial or failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run animal scan, character generation, story, and translation."
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs.yaml"),
        help="Shared YAML config.",
    )
    parser.add_argument(
        "--input-dir", help="Override configured image input directory."
    )
    parser.add_argument("--limit", type=int, help="Process at most this many images.")
    parser.add_argument("--support-species", help="Supporting animal species; missing, unsupported, or duplicate values fall back to deterministic random selection.")
    parser.add_argument("--support-color", help="Supporting animal color; missing, unsupported, or duplicate values fall back to deterministic random selection.")
    parser.add_argument("--areas", nargs="+", help="Area IDs for the story (choose at least two configured areas).")
    parser.add_argument("--generate-context-images", action="store_true", help="Generate a story-driven image for each story phase.")
    parser.add_argument("--back-translation", action="store_true", help="Back-translate saved translations to English for comparison (uses extra model requests).")
    parser.add_argument("--display", action="store_true", help="Open a local story reader after pipeline processing finishes.")
    parser.add_argument(
        "--refresh-speciesnet",
        action="store_true",
        help="Force SpeciesNet to refresh its cached model.",
    )
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        help="Reuse validated artifacts (default).",
    )
    resume.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Regenerate all pipeline stages.",
    )
    parser.set_defaults(resume=None)
    arguments = parser.parse_args(argv)
    if arguments.limit is not None and arguments.limit < 1:
        parser.error("--limit must be at least 1")
    configure_logging()
    config_path = Path(arguments.config).resolve()
    input_dir = Path(arguments.input_dir).resolve() if arguments.input_dir else None
    try:
        status = run_pipeline(
            config_path,
            input_dir,
            arguments.limit,
            arguments.refresh_speciesnet,
            arguments.resume,
            arguments.support_species,
            arguments.support_color,
            arguments.areas,
            arguments.generate_context_images,
            arguments.back_translation,
        )
        if arguments.display:
            with config_path.open(encoding="utf-8") as stream:
                display_config = yaml.safe_load(stream)
            output_root = Path(display_config["part_e"]["output_dir"])
            if not output_root.is_absolute():
                output_root = config_path.parent / output_root
            launch_streamlit_gallery(output_root.resolve(), config_path.parent)
        return status
    except (FileNotFoundError, ContractError, ValueError, yaml.YAMLError) as error:
        LOGGER.error("Pipeline setup failed: %s", error)
        return 2


if __name__ == "__main__":
    sys.exit(main())
