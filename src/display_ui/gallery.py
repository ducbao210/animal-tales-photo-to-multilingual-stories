"""Load completed story output and launch the Streamlit reader."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.contracts.schema import TARGET_LOCALES


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _validated_locale_artifact(
    path: Path, locale: str, expected_pages: list[int]
) -> dict[str, Any] | None:
    """Read one independently validated locale artifact when a bundle is absent or partial."""
    value = _read_json(path)
    if not value or value.get("locale") != locale:
        return None
    validation = value.get("validation")
    if not isinstance(validation, dict) or validation.get("passed") is not True:
        return None
    pages = value.get("pages")
    if (
        not isinstance(pages, list)
        or len(pages) != len(expected_pages)
        or any(not isinstance(page, dict) for page in pages)
        or [page.get("page") for page in pages] != expected_pages
    ):
        return None
    if any(
        not isinstance(page.get("text"), str) or not page["text"].strip()
        for page in pages
    ):
        return None
    return value


def _load_translations(
    photo_root: Path,
    bundle: dict[str, Any] | None,
    pages: list[dict[str, Any]],
) -> dict[str, Any]:
    """Load bundle translations and fill missing locales from validated per-locale files."""
    bundle_translations = bundle.get("translations", {}) if bundle else {}
    translations = dict(bundle_translations) if isinstance(bundle_translations, dict) else {}
    expected_pages = [page.get("page") for page in pages if isinstance(page, dict)]
    if len(expected_pages) != len(pages):
        return translations
    for locale in TARGET_LOCALES:
        if locale in translations:
            continue
        artifact = _validated_locale_artifact(
            photo_root / f"{locale}.json", locale, expected_pages
        )
        if artifact is not None:
            translations[locale] = artifact
    return translations


def load_story_records(output_root: str | Path, project_root: str | Path) -> list[dict[str, Any]]:
    """Load completed story bundles and resolve their input and illustration assets."""
    output_root = Path(output_root).resolve()
    project_root = Path(project_root).resolve()
    if not output_root.is_dir():
        return []
    records: list[dict[str, Any]] = []
    for photo_root in sorted(path for path in output_root.iterdir() if path.is_dir()):
        story = _read_json(photo_root / "story_en.json")
        scan = _read_json(photo_root / "scan.json")
        bundle = _read_json(photo_root / "translations.json")
        pages = story.get("pages") if story else None
        if not story or not isinstance(pages, list) or not pages or not scan:
            continue
        if story.get("image_id") != photo_root.name:
            continue
        translations = _load_translations(photo_root, bundle, pages)
        input_value = scan.get("input_path")
        input_path = Path(input_value) if isinstance(input_value, str) and input_value else None
        if input_path is not None and not input_path.is_absolute():
            input_path = (project_root / input_path).resolve()
        if input_path is not None and not input_path.is_file():
            input_path = None
        keyframe = photo_root / "character.png"
        phase_images: dict[str, Path] = {}
        for page in pages:
            if not isinstance(page, dict) or not isinstance(page.get("section"), str):
                continue
            section = page["section"]
            if section not in phase_images:
                generated = photo_root / f"phase_{section}.png"
                phase_images[section] = generated if generated.is_file() else keyframe
        records.append({
            "image_id": photo_root.name,
            "photo_root": photo_root,
            "input_path": input_path,
            "input_name": input_path.name if input_path else str(scan.get("input_path") or photo_root.name),
            "keyframe_path": keyframe if keyframe.is_file() else None,
            "phase_images": phase_images,
            "story": story,
            "translations": translations,
        })
    return records


def launch_streamlit_gallery(output_root: str | Path, project_root: str | Path) -> None:
    """Run the Streamlit reader against completed pipeline output."""
    app_path = Path(__file__).with_name("app.py").resolve()
    environment = os.environ.copy()
    environment["STORY_PIPELINE_OUTPUT_ROOT"] = str(Path(output_root).resolve())
    environment["STORY_PIPELINE_PROJECT_ROOT"] = str(Path(project_root).resolve())
    repository_root = str(Path(__file__).resolve().parents[2])
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (repository_root, existing_pythonpath) if part
    )
    command = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    subprocess.run(command, check=False, env=environment, cwd=repository_root)

