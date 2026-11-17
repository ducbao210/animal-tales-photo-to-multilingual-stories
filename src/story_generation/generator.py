"""Outline, draft, review, and bounded repair for eight-page stories."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.contracts.schema import SCHEMA_VERSION, CharacterResult, MainProfile, ScanResult, StageError, StoryPage, StoryResult, StoryWorld
from src.utils.files import atomic_write_json
from src.utils.retry import retry_provider_call
from .validation import validate_story_payload


OUTLINE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "problem": {"type": "string"},
        "area_sequence": {"type": "array", "items": {"type": "string"}},
        "move_reasons": {"type": "array", "items": {"type": "string"}},
        "main_skill_solution": {"type": "string"}, "support_skill_solution": {"type": "string"},
        "slogan_moments": {"type": "array", "items": {"type": "string"}},
        "four_parts": {"type": "object", "additionalProperties": False, "properties": {key: {"type": "string"} for key in ("beginning", "problem", "solution", "happy_ending")}, "required": ["beginning", "problem", "solution", "happy_ending"]},
    },
    "required": ["problem", "area_sequence", "move_reasons", "main_skill_solution", "support_skill_solution", "slogan_moments", "four_parts"],
}
STORY_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "title": {"type": "string"},
        "pages": {"type": "array", "items": {"type": "object", "additionalProperties": False,
            "properties": {"page": {"type": "integer"}, "section": {"type": "string"}, "area_id": {"type": "string"}, "text": {"type": "string"}},
            "required": ["page", "section", "area_id", "text"]}},
    },
    "required": ["title", "pages"],
}
REVIEW_SCHEMA = {"type": "object", "additionalProperties": False, "properties": {
    "passed": {"type": "boolean"}, "issues": {"type": "array", "items": {"type": "object", "additionalProperties": False, "properties": {
        "page": {"type": ["integer", "null"]}, "code": {"type": "string"}, "explanation": {"type": "string"}}, "required": ["page", "code", "explanation"]}},
}, "required": ["passed", "issues"]}

DEFAULT_PAGE_ILLUSTRATION_PROMPT = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "prompts"
    / "part_c"
    / "page_illustration.md"
)


def _load_template(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def _render(template: str, values: dict[str, Any]) -> str:
    import re
    fields = set(re.findall(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", template))
    if fields != set(values):
        raise ValueError(f"Prompt variable mismatch; expected {sorted(fields)}, received {sorted(values)}")
    return re.sub(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", lambda m: str(values[m.group(1)]), template)


def _call(client: Any, prompt: str, schema_name: str, schema: dict[str, Any], tokens: int, temperature: float,
          on_retry: Any = None) -> dict[str, Any]:
    response = retry_provider_call(
        lambda: client.generate(prompt, schema_name=schema_name, schema=schema, max_tokens=tokens, temperature=temperature),
        max_attempts=3, base_delay_seconds=1, on_retry=on_retry,
    )
    data = response.get("data")
    if not isinstance(data, dict):
        raise StageError("C", "invalid_response", False, "Story model returned no JSON object")
    return data


def choose_support(
    image_id: str, world: StoryWorld, main: MainProfile, main_color: str,
    allowed_colors: set[str], requested_species: str | None = None,
    requested_color: str | None = None,
) -> tuple[MainProfile, str]:
    species_choices = sorted(set(world.main) - {main.species})
    color_choices = sorted(allowed_colors - {main_color})
    if not species_choices or not color_choices:
        raise ValueError("Support requires at least one different configured species and color")
    if requested_species not in species_choices:
        digest = hashlib.sha256(f"{image_id}:support-species".encode("utf-8")).hexdigest()
        requested_species = species_choices[int(digest, 16) % len(species_choices)]
    if requested_color not in color_choices:
        digest = hashlib.sha256(f"{image_id}:support-color".encode("utf-8")).hexdigest()
        requested_color = color_choices[int(digest, 16) % len(color_choices)]
    return world.main[requested_species], requested_color


def generate_story(
    scan: ScanResult, character: CharacterResult, world: StoryWorld,
    settings: dict[str, Any], output_dir: Path, client: Any,
) -> StoryResult:
    if scan.status == "failed" or scan.species == "unknown" or scan.color is None:
        raise StageError("C", "unsupported_scan", False, "Story generation requires a supported species and color")
    if character.image_id != scan.image_id or character.species != scan.species or character.color != scan.color:
        raise StageError("C", "character_mismatch", False, "Character metadata does not match Stage 1")
    main = world.main[scan.species]
    support, support_color = choose_support(
        scan.image_id, world, main, scan.color, set(settings["allowed_colors"]),
        settings.get("support_species"), settings.get("support_color"),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_area_ids = set(settings.get("selected_areas") or world.areas_by_id())
    if len(selected_area_ids) < 2 or not selected_area_ids <= set(world.areas_by_id()):
        raise StageError("C", "invalid_areas", False, "Select at least two configured story areas")
    area_summary = [{"id": a["id"], "names": a["names"], "description": a["description"]} for a in world.location["areas"] if a["id"] in selected_area_ids]
    profiles = {"main": {**main.__dict__, "color": scan.color}, "support": {**support.__dict__, "color": support_color}}
    outline_template = _load_template(settings["prompt_outline"])
    outline_prompt = _render(outline_template, {
        "age": settings.get("age", "5-7"), "main_profile": json.dumps(profiles["main"], ensure_ascii=False),
        "support_profile": json.dumps(profiles["support"], ensure_ascii=False), "location": json.dumps(world.location["names"], ensure_ascii=False),
        "areas": json.dumps(area_summary, ensure_ascii=False), "species": scan.species, "color": scan.color,
    })
    on_retry = settings.get("on_retry")
    outline = _call(client, outline_prompt, "story_outline", OUTLINE_SCHEMA, int(settings.get("review_max_output_tokens", 700)), float(settings.get("temperature", 0.4)), on_retry)
    allowed_areas = selected_area_ids
    outline_issues = []
    sequence = outline.get("area_sequence", [])
    if not isinstance(sequence, list) or len(set(sequence) & allowed_areas) < 2 or any(item not in allowed_areas for item in sequence):
        outline_issues.append("area_sequence must name at least two supplied areas")
    if not isinstance(outline.get("move_reasons"), list) or not outline["move_reasons"]:
        outline_issues.append("move_reasons is required")
    if not outline.get("main_skill_solution") or not outline.get("support_skill_solution"):
        outline_issues.append("both characters must use their skills causally")
    if outline_issues:
        raise StageError("C", "invalid_outline", False, "; ".join(outline_issues))
    atomic_write_json(output_dir / "outline.json", outline)

    write_template = _load_template(settings["prompt_write"])
    write_values = {
        "age": settings.get("age", "5-7"),
        "outline": json.dumps(outline, ensure_ascii=False), "main_profile": json.dumps(profiles["main"], ensure_ascii=False),
        "support_profile": json.dumps(profiles["support"], ensure_ascii=False),
        "character_metadata": json.dumps({"name": character.name, "species": character.species, "color": character.color, "image_path": character.image_path}, ensure_ascii=False),
        "area_names": json.dumps(area_summary, ensure_ascii=False), "page_limit": 40,
        "species": scan.species, "color": scan.color,
    }
    draft_prompt = _render(write_template, write_values)
    prompt_digest = hashlib.sha256((outline_prompt + draft_prompt).encode("utf-8")).hexdigest()
    payload = _call(client, draft_prompt, "english_story", STORY_SCHEMA, int(settings.get("max_output_tokens", 1500)), float(settings.get("temperature", 0.4)), on_retry)
    reviews: list[dict[str, Any]] = []
    review_template = _load_template(settings["prompt_review"])
    refine_template = _load_template(settings["prompt_refine"])
    rounds = 0
    max_rounds = int(settings.get("correction_rounds", 2))
    while True:
        deterministic = validate_story_payload(payload, image_id=scan.image_id, species=scan.species, color=scan.color,
            main=main, support=support, support_color=support_color, world=world, allowed_areas=allowed_areas)
        review_prompt = _render(review_template, {
            "story": json.dumps(payload, ensure_ascii=False), "profiles": json.dumps(profiles, ensure_ascii=False),
            "areas": json.dumps(area_summary, ensure_ascii=False),
            "age": settings.get("age", "5-7"),
        })
        model_review = _call(client, review_prompt, "story_review", REVIEW_SCHEMA, int(settings.get("review_max_output_tokens", 700)), 0, on_retry)
        issues = deterministic + list(model_review.get("issues", []))
        review = {"passed": not deterministic and bool(model_review.get("passed")), "issues": issues, "round": rounds}
        reviews.append(review)
        atomic_write_json(output_dir / f"story_review_r{rounds}.json", review)
        atomic_write_json(output_dir / f"story_draft_r{rounds}.json", payload)
        if review["passed"]:
            break
        if rounds >= max_rounds:
            raise StageError("C", "validation_exhausted", False, f"Story failed validation after {rounds} correction rounds")
        refine_prompt = _render(refine_template, {
            "draft": json.dumps(payload, ensure_ascii=False), "issues": json.dumps(issues, ensure_ascii=False),
            "glossary": json.dumps(profiles | {"areas": area_summary}, ensure_ascii=False),
            "age": settings.get("age", "5-7"),
        })
        rounds += 1
        payload = _call(client, refine_prompt, "refined_story", STORY_SCHEMA, int(settings.get("max_output_tokens", 1500)), float(settings.get("temperature", 0.4)), on_retry)

    pages = tuple(StoryPage(page=item["page"], section=item["section"], area_id=item["area_id"], text=item["text"]) for item in payload["pages"])
    result = StoryResult(
        schema_version=SCHEMA_VERSION, image_id=scan.image_id, title=payload["title"], language="en",
        character_id=main.species, species=scan.species, color=scan.color, support_id=support.species, support_color=support_color,
        location_id=str(world.location.get("id", "sunny_garden")), pages=pages,
        validation={"passed": True, "issues": [], "rounds": rounds}, model=str(settings["model_id"]), prompt_sha256=prompt_digest,
    )
    atomic_write_json(output_dir / "story_en.json", result.to_dict())
    # Keep the required page-specific prompts as text artifacts; image calls remain an explicit optional feature.
    page_template = _load_template(
        settings.get(
            "prompt_page_illustration", DEFAULT_PAGE_ILLUSTRATION_PROMPT
        )
    )
    prompts = []
    for page in pages:
        support_present = support.name.casefold() in page.text.casefold()
        roster = {
            main.species: {"name": main.name, "species": main.species, "color": scan.color, "visual_detail": main.visual_detail},
        }
        present = {main.species: roster[main.species]}
        if support_present:
            roster[support.species] = {"name": support.name, "species": support.species, "color": support_color, "visual_detail": support.visual_detail}
            present[support.species] = roster[support.species]
        prompts.append({"page": page.page, "prompt": _render(page_template, {
            "page_text": page.text, "page_id": page.page,
            "reference_description": f"{scan.color} {scan.species}, {main.visual_detail}",
            "characters": json.dumps(roster, ensure_ascii=False),
            "characters_present": json.dumps(present, ensure_ascii=False),
            "pose": "as described by the page", "emotion": "as described by the page",
            "area": world.areas_by_id()[page.area_id]["names"]["en"],
        })})
    atomic_write_json(output_dir / "page_illustration_prompts.json", prompts)
    return result
