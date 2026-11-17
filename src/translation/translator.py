"""Translate, review, repair, and persist target-language stories."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.contracts.schema import SCHEMA_VERSION, TARGET_LOCALES, StageError, StoryResult, StoryWorld, Translation, TranslationBundle
from src.utils.files import atomic_write_json
from src.utils.retry import retry_provider_call
from .validation import validate_translation_payload


TRANSLATION_SCHEMA = {"type": "object", "additionalProperties": False, "properties": {
    "pages": {"type": "array", "items": {"type": "object", "additionalProperties": False,
        "properties": {"page": {"type": "integer"}, "text": {"type": "string"}}, "required": ["page", "text"]}},
}, "required": ["pages"]}
REVIEW_SCHEMA = {"type": "object", "additionalProperties": False, "properties": {
    "passed": {"type": "boolean"}, "issues": {"type": "array", "items": {"type": "object", "additionalProperties": False,
        "properties": {"page": {"type": ["integer", "null"]}, "group": {"type": "string"}, "code": {"type": "string"}, "explanation": {"type": "string"}},
        "required": ["page", "group", "code", "explanation"]}},
}, "required": ["passed", "issues"]}


def _render(template: str, values: dict[str, Any]) -> str:
    import re
    fields = set(re.findall(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", template))
    if fields != set(values):
        raise ValueError(f"Prompt variable mismatch; expected {sorted(fields)}, received {sorted(values)}")
    return re.sub(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", lambda m: str(values[m.group(1)]), template)


def _call(client: Any, prompt: str, schema_name: str, schema: dict[str, Any], tokens: int, temperature: float,
          on_retry: Any = None, locale: str | None = None, on_event: Any = None) -> dict[str, Any]:
    try:
        result = retry_provider_call(
            lambda: client.generate(prompt, schema_name=schema_name, schema=schema, max_tokens=tokens, temperature=temperature),
            max_attempts=3, base_delay_seconds=1, on_retry=on_retry,
        )
    except StageError as error:
        if on_event:
            on_event({"locale": locale, "event": "step_end", "status": "failed", "category": error.category, "message": error.message})
        if locale:
            raise StageError("D", error.category, error.retryable, f"{locale}: {error.message}") from error
        raise
    data = result.get("data")
    if not isinstance(data, dict):
        raise StageError("D", "invalid_response", False, "Translation model returned no JSON object")
    return data


def translate_story(
    story: StoryResult, world: StoryWorld, settings: dict[str, Any], output_dir: Path,
    client: Any, reviewer_client: Any | None = None,
    cache_fingerprint: str | None = None,
    reuse_cache: bool = True,
) -> TranslationBundle:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_pages = [{"page": page.page, "text": page.text} for page in story.pages]
    if len(source_pages) != 8 or [p["page"] for p in source_pages] != list(range(1, 9)):
        raise StageError("D", "invalid_source", False, "Accepted English story does not have eight aligned pages")
    encoded = json.dumps(story.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    source_hash = hashlib.sha256(encoded).hexdigest()
    locales = tuple(settings.get("locales", TARGET_LOCALES))
    if set(locales) != set(TARGET_LOCALES):
        raise StageError("D", "invalid_config", False, "Translation locales must be fr-FR, es-ES, and vi-VN")
    all_translations: dict[str, Translation] = {}
    reviewer_client = reviewer_client or client
    names_by_locale = {
        locale: {species: profile.names[locale] for species, profile in world.main.items()}
        for locale in TARGET_LOCALES
    }
    all_area_names = {area_id: area["names"] for area_id, area in world.areas_by_id().items()}
    required_area_names = {
        area["names"]["en"]: area["names"]
        for area in world.location["areas"]
    }
    profiles = {"main": world.main[story.species], "support": world.main[story.support_id]}
    glossary_common = {
        "genders": {key: item.gender for key, item in profiles.items()},
        "species": {key: item.species for key, item in profiles.items()},
        "colors": {"main": story.color, "support": story.support_color},
        "pronouns": world.pronouns,
        "slogans": {key: item.slogans for key, item in profiles.items()},
        "areas": all_area_names,
    }
    translate_template = Path(settings["prompt_translate"]).read_text(encoding="utf-8")
    review_template = Path(settings["prompt_review"]).read_text(encoding="utf-8")
    refine_template = Path(settings["prompt_refine"]).read_text(encoding="utf-8")
    on_retry = settings.get("on_retry")
    on_event = settings.get("on_event")
    locale_failures: list[tuple[str, StageError]] = []
    for locale in locales:
        try:
            if on_event:
                on_event({"locale": locale, "event": "step_start", "status": "running"})
            existing_path = output_dir / f"{locale}.json"
            required_names = {
                profile.names["en"]: profile.names[locale]
                for profile in (world.main[story.species], world.main[story.support_id])
            }
            required_areas = {source: names[locale] for source, names in required_area_names.items()}
            cached_translation = None
            if reuse_cache and existing_path.is_file():
                try:
                    old = json.loads(existing_path.read_text(encoding="utf-8"))
                    if (
                        old.get("source_story_sha256") == source_hash
                        and old.get("target_age", "5-7") == settings.get("age", "5-7")
                        and (
                            cache_fingerprint is None
                            or old.get("translation_fingerprint") == cache_fingerprint
                        )
                        and old.get("validation", {}).get("passed")
                    ):
                        candidate = {"pages": old["pages"]}
                        candidate_issues = validate_translation_payload(
                            source_pages, candidate, locale=locale,
                            required_names=required_names, required_areas=required_areas,
                        )
                        if not candidate_issues:
                            cached_translation = Translation(locale=locale, pages=tuple(candidate["pages"]), validation=old["validation"])
                    else:
                        cached_translation = None
                except (OSError, ValueError, KeyError):
                    cached_translation = None
            else:
                cached_translation = None
            if cached_translation is not None:
                all_translations[locale] = cached_translation
                if on_event:
                    on_event({"locale": locale, "event": "step_end", "status": "skipped", "reason": "validated locale artifact"})
                continue
            payload = None
            values = {
                "source_story": json.dumps(story.to_dict(), ensure_ascii=False), "locale": locale,
                "age": settings.get("age", "5-7"),
                "names": json.dumps(names_by_locale[locale], ensure_ascii=False),
                "genders": json.dumps(glossary_common["genders"], ensure_ascii=False),
                "pronouns": json.dumps(world.pronouns.get(locale, {}), ensure_ascii=False),
                "slogans": json.dumps({key: profile.slogans[locale] for key, profile in profiles.items()}, ensure_ascii=False),
                "areas": json.dumps({key: value[locale] for key, value in all_area_names.items()}, ensure_ascii=False),
                "punctuation_notes": world.punctuation[locale], "examples": {
                    "fr-FR": "Source: Milo said, 'I can help!' | Example: Milo dit : « Je peux aider ! »",
                    "es-ES": "Source: Milo said, 'I can help!' | Example: Milo dijo: «¡Puedo ayudar!»",
                    "vi-VN": "Source: Pip the hare said, 'I can help!' | Example: Thỏ Pip nói: “Mình có thể giúp!”",
                }[locale] + " Use only as language-format guidance; do not copy sample story content.",
            }
            prompt = _render(translate_template, values)
            if payload is None:
                payload = _call(client, prompt, f"translation_{locale.replace('-', '_')}", TRANSLATION_SCHEMA,
                    int(settings.get("max_completion_tokens", 1500)), float(settings.get("temperature", 0.2)), on_retry, locale, on_event)
            rounds = 0
            locale_reviews = []
            while True:
                local_issues = validate_translation_payload(
                    source_pages, payload, locale=locale, required_names=required_names,
                    required_areas=required_areas,
                )
                glossary = {**glossary_common, "names": names_by_locale[locale], "locale": locale}
                review_prompt = _render(review_template, {
                    "source_story": json.dumps(story.to_dict(), ensure_ascii=False),
                    "translation": json.dumps(payload, ensure_ascii=False), "locale": locale,
                    "age": settings.get("age", "5-7"),
                    "glossary": json.dumps(glossary, ensure_ascii=False),
                })
                review = _call(reviewer_client, review_prompt, f"translation_review_{locale.replace('-', '_')}", REVIEW_SCHEMA,
                    int(settings.get("review_max_completion_tokens", 800)), 0, on_retry, locale, on_event)
                issues = local_issues + list(review.get("issues", []))
                passed = not local_issues and bool(review.get("passed"))
                locale_reviews.append({"passed": passed, "issues": issues, "round": rounds})
                atomic_write_json(output_dir / f"translation_{locale}_r{rounds}.json", payload)
                atomic_write_json(output_dir / f"translation_review_{locale}_r{rounds}.json", locale_reviews[-1])
                if passed:
                    break
                if rounds >= int(settings.get("correction_rounds", 2)):
                    raise StageError("D", "validation_exhausted", False, f"Translation {locale} failed review after {rounds} correction rounds")
                refine_prompt = _render(refine_template, {
                    "translation": json.dumps(payload, ensure_ascii=False),
                    "source_story": json.dumps(story.to_dict(), ensure_ascii=False),
                    "issues": json.dumps(issues, ensure_ascii=False), "glossary": json.dumps(glossary, ensure_ascii=False),
                    "age": settings.get("age", "5-7"),
                })
                rounds += 1
                payload = _call(client, refine_prompt, f"refined_{locale.replace('-', '_')}", TRANSLATION_SCHEMA,
                    int(settings.get("max_completion_tokens", 1500)), float(settings.get("temperature", 0.2)), on_retry, locale, on_event)
            translation = Translation(locale=locale, pages=tuple(payload["pages"]), validation={"passed": True, "issues": [], "rounds": rounds})
            all_translations[locale] = translation
            atomic_write_json(existing_path, {
                "schema_version": SCHEMA_VERSION, "image_id": story.image_id,
                "source_story_sha256": source_hash, "target_age": settings.get("age", "5-7"),
                **({"translation_fingerprint": cache_fingerprint} if cache_fingerprint is not None else {}),
                **translation.to_dict(),
            })
            if on_event:
                on_event({"locale": locale, "event": "step_end", "status": "succeeded"})
        except StageError as error:
            if error.category in {
                "invalid_config", "invalid_source", "missing_credentials", "dependency_missing",
            }:
                raise
            locale_failures.append((locale, error))
            if on_event:
                on_event({
                    "locale": locale, "event": "step_end", "status": "failed",
                    "category": error.category, "message": error.message,
                })
    bundle = TranslationBundle(SCHEMA_VERSION, story.image_id, source_hash, all_translations,
        {"passed": len(all_translations) == 3, "locales": list(all_translations),
         "failed_locales": {locale: error.category for locale, error in locale_failures}})
    atomic_write_json(output_dir / "translations.json", bundle.to_dict())
    if locale_failures:
        failures = ", ".join(f"{locale} ({error.category})" for locale, error in locale_failures)
        raise StageError("D", "partial_translation", False, f"Unfinished translations: {failures}")
    return bundle


def back_translate_story(
    story: StoryResult, settings: dict[str, Any], output_dir: Path, client: Any,
) -> dict[str, dict[str, Any]]:
    """Back-translate saved locale artifacts for an explicit diagnostic run."""
    template = Path(settings["prompt_back_translation"]).read_text(encoding="utf-8")
    back_schema = {"type": "object", "additionalProperties": False, "properties": {
        "pages": TRANSLATION_SCHEMA["properties"]["pages"],
        "differences": {"type": "array", "items": {"type": "object", "additionalProperties": False,
            "properties": {"page": {"type": "integer"}, "type": {"type": "string"}, "explanation": {"type": "string"}},
            "required": ["page", "type", "explanation"]}},
    }, "required": ["pages", "differences"]}
    source_story = json.dumps(story.to_dict(), ensure_ascii=False)
    results: dict[str, dict[str, Any]] = {}
    for locale in settings.get("locales", TARGET_LOCALES):
        translation_path = output_dir / f"{locale}.json"
        try:
            translation_data = json.loads(translation_path.read_text(encoding="utf-8"))
            pages = translation_data["pages"]
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise StageError("D_back_translation", "invalid_translation", False,
                             f"Cannot back-translate missing or invalid {locale} output") from error
        translation = Translation(
            locale=locale,
            pages=tuple(pages),
            validation=translation_data.get("validation", {}),
        )
        prompt = _render(template, {
            "translation": json.dumps(translation.to_dict(), ensure_ascii=False),
            "locale": locale,
            "source_story": source_story,
        })
        try:
            payload = _call(
                client, prompt, f"back_translation_{locale.replace('-', '_')}", back_schema,
                int(settings.get("max_completion_tokens", 1500)),
                float(settings.get("temperature", 0.2)), settings.get("on_retry"), locale,
                settings.get("on_event"),
            )
        except StageError as error:
            raise StageError(
                "D_back_translation", error.category, error.retryable,
                f"{locale}: {error.message}",
            ) from error
        atomic_write_json(output_dir / f"back_translation_{locale}.json", payload)
        results[locale] = payload
    return results
