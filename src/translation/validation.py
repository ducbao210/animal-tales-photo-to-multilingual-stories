"""Deterministic alignment and glossary checks for translated pages."""
from __future__ import annotations

import re
import unicodedata
from typing import Any


def _fold(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


def validate_translation_payload(
    source_pages: list[dict[str, Any]], payload: dict[str, Any], *, locale: str,
    required_names: dict[str, str], required_areas: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    pages = payload.get("pages")
    if not isinstance(pages, list) or len(pages) != len(source_pages):
        return [{"page": None, "group": "consistency", "code": "page_count", "explanation": "Translation page count must match the source."}]
    for source, translated, page_number in zip(source_pages, pages, range(1, len(source_pages) + 1)):
        if not isinstance(translated, dict) or translated.get("page") != source.get("page"):
            issues.append({"page": page_number, "group": "consistency", "code": "page_id", "explanation": "Translation page IDs must exactly match the source."})
            continue
        text = translated.get("text")
        if not isinstance(text, str) or not text.strip():
            issues.append({"page": page_number, "group": "consistency", "code": "empty_page", "explanation": "Translated page must not be empty."})
            continue
        folded = _fold(text)
        source_text = _fold(str(source.get("text", "")))
        for source_name, translated_name in required_names.items():
            if _fold(source_name) in source_text and _fold(translated_name) not in folded:
                issues.append({"page": page_number, "group": "consistency", "code": "missing_name", "explanation": f"Required mapped name {source_name} ({translated_name}) is missing."})
        for source_area, translated_area in (required_areas or {}).items():
            if _fold(source_area) in source_text and _fold(translated_area) not in folded:
                issues.append({"page": page_number, "group": "consistency", "code": "area_name", "explanation": f"Area name {source_area} must use its mapped form {translated_area}."})
        if locale == "es-ES":
            if "?" in text and "¿" not in text:
                issues.append({"page": page_number, "group": "consistency", "code": "spanish_question_marks", "explanation": "Spanish questions need an opening ¿ mark."})
            if "!" in text and "¡" not in text:
                issues.append({"page": page_number, "group": "consistency", "code": "spanish_exclamation_marks", "explanation": "Spanish exclamations need an opening ¡ mark."})
        if locale == "fr-FR" and re.search(r"\S[!?;:]", text):
            issues.append({"page": page_number, "group": "consistency", "code": "french_spacing", "explanation": "French typography needs a space before ! ? : ;."})
    return issues
