"""Preflight validation for shared YAML settings and external prerequisites."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from src.contracts.schema import ContractError, TARGET_LOCALES, StoryWorld


def load_pipeline_config(path: str | Path) -> tuple[Path, dict[str, Any], StoryWorld]:
    config_path = Path(path).resolve()
    root = config_path.parent
    # Prefer project credentials over stale values inherited from the shell.
    load_dotenv(root / ".env", override=True)
    with config_path.open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict):
        raise ContractError("Configuration must be a YAML mapping")
    for key in ("part_a", "part_b", "part_c", "part_d", "part_e"):
        if not isinstance(document.get(key), dict):
            raise ContractError(f"Configuration must define {key} as a mapping")
    species = set(document["part_a"]["taxonomy"]["species"])
    colors = set(document["part_a"]["taxonomy"]["colors"])
    if species != {"cat", "owl", "hare", "dog", "squirrel"}:
        raise ContractError("Configured species must be cat, owl, hare, dog, and squirrel")
    if colors != {"black", "white", "brown", "orange", "gray"}:
        raise ContractError("Configured colors must be black, white, brown, orange, and gray")
    part_b, part_c, part_d, part_e = (document[key] for key in ("part_b", "part_c", "part_d", "part_e"))
    if part_b.get("backend") != "replicate":
        raise ContractError("part_b.backend must be replicate")
    for section_name in ("part_b", "part_c", "part_d"):
        model_id = document[section_name].get("model_id")
        if not isinstance(model_id, str) or not model_id.strip():
            raise ContractError(f"{section_name}.model_id must be a non-empty string")
    if part_d.get("locales") != list(TARGET_LOCALES):
        raise ContractError("part_d.locales must be [fr-FR, es-ES, vi-VN]")
    if not 0 <= float(part_c.get("temperature", -1)) <= 2:
        raise ContractError("part_c.temperature must be between 0 and 2")
    if not 0 <= float(part_d.get("temperature", -1)) <= 2:
        raise ContractError("part_d.temperature must be between 0 and 2")
    if int(part_e.get("max_attempts", 0)) < 1:
        raise ContractError("part_e.max_attempts must be at least one")
    world_path = root / part_e["world_path"]
    for key, section in (("prompt_path", part_b),
                         ("prompt_outline", part_c), ("prompt_write", part_c),
                         ("prompt_review", part_c), ("prompt_refine", part_c),
                         ("prompt_phase_plan", part_c),
                         ("prompt_page_illustration", part_c),
                         ("prompt_context", part_c),
                         ("prompt_translate", part_d), ("prompt_review", part_d), ("prompt_refine", part_d)):
        if not (root / section[key]).is_file():
            raise ContractError(f"Configured prompt file does not exist: {section[key]}")
    if not (root / part_d["prompt_back_translation"]).is_file():
        raise ContractError(
            f"Configured prompt file does not exist: {part_d['prompt_back_translation']}"
        )
    if not world_path.is_file():
        raise ContractError("Story world YAML file must exist")
    part_e["world_path"] = str(world_path.resolve())
    with world_path.open(encoding="utf-8") as stream:
        world_doc = yaml.safe_load(stream)
    world = StoryWorld.from_dict(world_doc, species)
    needed = ["GEMINI_API_KEY", "REPLICATE_API_TOKEN", "GROQ_API_KEY"]
    missing = [name for name in needed if not os.environ.get(name)]
    if missing:
        raise ContractError("Missing required environment variable(s): " + ", ".join(missing))
    for section in (part_b, part_c, part_d):
        for key in list(section):
            if key.endswith("_path") or key.startswith("prompt_"):
                section[key] = str((root / section[key]).resolve())
    for key in ("output_dir", "log_dir"):
        part_e[key] = str((root / part_e[key]).resolve())
    return root, document, world
