"""Local story checks that complement the model's safety and narrative review."""
from __future__ import annotations

import re
import unicodedata
from typing import Any

from src.contracts.schema import MainProfile, StoryWorld


def _fold(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


def _contains_whole_phrase(text: str, phrase: str) -> bool:
    return re.search(r"(?<!\w)" + re.escape(_fold(phrase)) + r"(?!\w)", _fold(text), re.UNICODE) is not None


def validate_story_payload(
    payload: dict[str, Any], *, image_id: str, species: str, color: str,
    main: MainProfile, support: MainProfile, support_color: str, world: StoryWorld,
    allowed_areas: set[str],
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    pages = payload.get("pages")
    if not isinstance(pages, list) or len(pages) != 8:
        return [{"page": None, "code": "page_count", "explanation": "Story must contain exactly eight pages."}]
    expected_sections = [section for section in ("beginning", "problem", "solution", "happy_ending") for _ in range(2)]
    area_ids = world.areas_by_id()
    used_areas: set[str] = set()
    text_all: list[str] = []
    for idx, page in enumerate(pages, start=1):
        if not isinstance(page, dict):
            issues.append({"page": idx, "code": "page_type", "explanation": "Page must be an object."})
            continue
        if page.get("page") != idx:
            issues.append({"page": idx, "code": "page_id", "explanation": "Page IDs must be consecutive from 1 to 8."})
        if page.get("section") != expected_sections[idx - 1]:
            issues.append({"page": idx, "code": "section_order", "explanation": "Each story section must contain its assigned two pages."})
        area_id = page.get("area_id")
        if area_id not in area_ids or area_id not in allowed_areas:
            issues.append({"page": idx, "code": "unknown_area", "explanation": "Page references an unknown garden area."})
        else:
            used_areas.add(area_id)
        text = page.get("text")
        if not isinstance(text, str) or not text.strip():
            issues.append({"page": idx, "code": "empty_page", "explanation": "Page text must be non-empty."})
            continue
        text_all.append(text)
        words = len(text.split())
        if words > 40:
            issues.append({"page": idx, "code": "word_limit", "explanation": f"Page has {words} words; limit is 40."})
    if len(used_areas) < 2:
        issues.append({"page": None, "code": "area_variety", "explanation": "The story must visit at least two distinct areas."})
    full_text = "\n".join(text_all)
    for label, phrase in (("main_name", main.name), ("support_name", support.name), ("species", species), ("color", color), ("support_species", support.species), ("support_color", support_color)):
        if not _contains_whole_phrase(full_text, phrase):
            issues.append({"page": None, "code": f"missing_{label}", "explanation": f"Story must preserve the main {label.replace('_', ' ')}."})
    for profile_label, slogan, profile_name in (("main", main.slogan_en, main.name), ("support", support.slogan_en, support.name)):
        spoken = any(_contains_whole_phrase(text, slogan) and profile_name in text and any(mark in text for mark in ('"', "“", "‘", "'")) for text in text_all)
        if not spoken:
            issues.append({"page": None, "code": f"slogan_{profile_label}", "explanation": f"{profile_label.title()} character's slogan must be spoken in dialogue."})
    # Catch a third named character drawn from the known profile roster. Semantic review handles unnamed actors.
    known_names = {profile.name for profile in world.main.values()}
    expected_names = {main.name, support.name}
    for name in known_names - expected_names:
        if _contains_whole_phrase(full_text, name):
            issues.append({"page": None, "code": "third_character", "explanation": "Story includes another configured character."})
            break
    return issues
