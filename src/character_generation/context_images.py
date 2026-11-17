"""Optional story-driven illustrations for the four narrative phases."""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from src.contracts.schema import SECTIONS, StageError, StoryResult, StoryWorld, ensure_under
from src.utils.files import atomic_write_bytes, atomic_write_json
from src.story_generation.generator import _call, _load_template, _render


PHASE_PLAN_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"plans": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "properties": {
            "section": {"type": "string"}, "page_ids": {"type": "array", "items": {"type": "integer"}},
            "area_id": {"type": "string"}, "scene": {"type": "string"},
            "action": {"type": "string"}, "emotion": {"type": "string"},
            "character_ids": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["section", "page_ids", "area_id", "scene", "action", "emotion", "character_ids"],
    }}}, "required": ["plans"],
}


def validate_phase_plans(
    plans: Any, story: StoryResult, allowed_areas: set[str], character_ids: set[str],
) -> list[dict[str, Any]]:
    if not isinstance(plans, list) or len(plans) != len(SECTIONS):
        raise StageError("C_images", "invalid_phase_plan", False, "Visual planner must return exactly four phase plans")
    expected_pages = {section: [page.page for page in story.pages if page.section == section] for section in SECTIONS}
    seen: set[str] = set()
    for plan in plans:
        if not isinstance(plan, dict):
            raise StageError("C_images", "invalid_phase_plan", False, "Each visual plan must be an object")
        section = plan.get("section")
        if section not in SECTIONS or section in seen:
            raise StageError("C_images", "invalid_phase_plan", False, "Visual plans must contain each story phase exactly once")
        seen.add(section)
        if plan.get("page_ids") != expected_pages[section]:
            raise StageError("C_images", "invalid_phase_plan", False, f"Page IDs do not match phase {section}")
        phase_areas = {page.area_id for page in story.pages if page.section == section}
        if plan.get("area_id") not in allowed_areas or plan.get("area_id") not in phase_areas:
            raise StageError("C_images", "invalid_phase_plan", False, f"Phase {section} uses an unselected area")
        ids = plan.get("character_ids")
        if not isinstance(ids, list) or not ids or len(ids) > 2 or len(set(ids)) != len(ids) or not set(ids) <= character_ids:
            raise StageError("C_images", "invalid_phase_plan", False, f"Phase {section} contains an unknown or duplicate character")
        if story.species not in ids:
            raise StageError("C_images", "invalid_phase_plan", False, f"Phase {section} must include the main character")
        for key in ("scene", "action", "emotion"):
            if not isinstance(plan.get(key), str) or not plan[key].strip():
                raise StageError("C_images", "invalid_phase_plan", False, f"Phase {section} is missing {key}")
    return sorted(plans, key=lambda plan: SECTIONS.index(plan["section"]))


def generate_context_images(
    story: StoryResult, world: StoryWorld, story_settings: dict[str, Any],
    image_settings: dict[str, Any], output_dir: Path, story_client: Any, image_client: Any,
) -> list[Path]:
    main = world.main[story.species]
    support = world.main[story.support_id]
    selected_areas = set(story_settings.get("selected_areas") or world.areas_by_id())
    area_summary = [
        {"id": area_id, "name": world.areas_by_id()[area_id]["names"]["en"],
         "description": world.areas_by_id()[area_id]["description"]}
        for area_id in sorted(selected_areas)
    ]
    roster = {
        main.species: {"name": main.name, "species": main.species, "color": story.color, "visual_detail": main.visual_detail},
        support.species: {"name": support.name, "species": support.species, "color": story.support_color, "visual_detail": support.visual_detail},
    }
    pages_by_phase = {
        section: [{"page": page.page, "area_id": page.area_id, "text": page.text}
                  for page in story.pages if page.section == section]
        for section in SECTIONS
    }
    plan_template = _load_template(story_settings["prompt_phase_plan"])
    plan_prompt = _render(plan_template, {
        "story_pages": json.dumps(pages_by_phase, ensure_ascii=False),
        "characters": json.dumps(roster, ensure_ascii=False),
        "areas": json.dumps(area_summary, ensure_ascii=False),
    })
    response = _call(story_client, plan_prompt, "story_phase_visual_plan", PHASE_PLAN_SCHEMA,
                     int(story_settings.get("review_max_output_tokens", 700)), 0,
                     story_settings.get("on_retry"))
    plans = validate_phase_plans(response.get("plans"), story, selected_areas, set(roster))
    atomic_write_json(output_dir / "phase_visual_plan.json", {"plans": plans})
    template = _load_template(story_settings["prompt_context"])
    area_lookup = world.areas_by_id()
    outputs: list[Path] = []
    rendered_prompts: list[dict[str, Any]] = []
    for plan in plans:
        phase_pages = pages_by_phase[plan["section"]]
        present = {key: roster[key] for key in plan["character_ids"]}
        prompt = _render(template, {
            "section": plan["section"],
            "phase_text": json.dumps(phase_pages, ensure_ascii=False),
            "visual_plan": json.dumps(plan, ensure_ascii=False),
            "characters": json.dumps(roster, ensure_ascii=False),
            "characters_present": json.dumps(present, ensure_ascii=False),
            "area": json.dumps(area_lookup[plan["area_id"]]["names"]["en"], ensure_ascii=False),
            "consistency_guidance": "Maintain the same illustration style and reconstruct every character using only the fixed appearance metadata included in this prompt.",
        })
        rendered_prompts.append({"section": plan["section"], "prompt": prompt})
        path = ensure_under(output_dir / f"phase_{plan['section']}.png", output_dir)
        settings = {
            key: value for key, value in image_settings.items()
            if key not in {"reference_image_path", "reference_image_paths"}
        }
        response = image_client.generate(prompt, output_dir=output_dir, settings=settings)
        try:
            with Image.open(io.BytesIO(response.data)) as image:
                image.verify()
            with Image.open(io.BytesIO(response.data)) as image:
                normalized = io.BytesIO()
                image.convert("RGBA").save(normalized, format="PNG", optimize=True)
        except (OSError, UnidentifiedImageError, AttributeError) as error:
            raise StageError("C_images", "image_decode", False, f"Phase {plan['section']} image is invalid") from error
        atomic_write_bytes(path, normalized.getvalue())
        outputs.append(path)
    atomic_write_json(output_dir / "phase_image_prompts.json", rendered_prompts)
    return outputs
