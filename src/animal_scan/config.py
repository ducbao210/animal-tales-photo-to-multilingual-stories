from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PartAConfig:
    project_root: Path
    values: dict[str, Any]

    @property
    def taxonomy(self) -> dict[str, Any]:
        return self.values["taxonomy"]

    @property
    def paths(self) -> dict[str, Path]:
        return {
            key: resolve_path(self.project_root, value)
            for key, value in self.values["paths"].items()
        }


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_config_document(path: str | Path) -> tuple[Path, dict[str, Any]]:
    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict):
        raise ValueError("Configuration must contain a mapping at its root")
    return config_path.parent, document


def load_config(path: str | Path) -> PartAConfig:
    project_root, document = load_config_document(path)
    if not isinstance(document.get("part_a"), dict):
        raise ValueError("Configuration must contain a 'part_a' mapping")

    values = document["part_a"]
    taxonomy = values.get("taxonomy", {})
    if not isinstance(taxonomy, dict):
        raise ValueError("part_a.taxonomy must be a mapping")
    species = taxonomy.get("species")
    colors = taxonomy.get("colors")
    for name, labels in (("species", species), ("colors", colors)):
        if (
            not isinstance(labels, list)
            or not labels
            or not all(isinstance(label, str) and label.strip() for label in labels)
            or len(set(labels)) != len(labels)
        ):
            raise ValueError(f"taxonomy.{name} must contain unique non-empty labels")
    unknown_species = taxonomy.get("unknown_species")
    if (
        not isinstance(unknown_species, str)
        or not unknown_species.strip()
        or unknown_species in species
    ):
        raise ValueError("taxonomy.unknown_species must be a non-empty label")

    aliases = taxonomy.get("species_aliases", {})
    if not isinstance(aliases, dict) or set(aliases) != set(species):
        raise ValueError("species_aliases must define every configured species exactly once")
    for species_name, accepted_labels in aliases.items():
        if not isinstance(accepted_labels, list) or not accepted_labels or not all(
            isinstance(alias, str) and alias.strip() for alias in accepted_labels
        ):
            raise ValueError(f"species_aliases.{species_name} must contain labels")

    retries = values.get("retries", {})
    if int(retries.get("max_attempts", 0)) < 1:
        raise ValueError("retries.max_attempts must be at least 1")
    if float(retries.get("retry_delay_seconds", -1)) < 0:
        raise ValueError("retries.retry_delay_seconds cannot be negative")
    concurrency = values.get("concurrency", {})
    if int(concurrency.get("max_workers", 0)) < 1:
        raise ValueError("concurrency.max_workers must be at least 1")
    clip_settings = values.get("clip", {})
    if not isinstance(clip_settings, dict):
        raise ValueError("part_a.clip must be a mapping")
    prompts = clip_settings.get("prompts", {})
    if (
        not isinstance(prompts, dict)
        or set(prompts) != set(colors)
        or not all(isinstance(prompt, str) and prompt.strip() for prompt in prompts.values())
    ):
        raise ValueError("clip.prompts must define every configured color exactly once")
    margin = float(clip_settings.get("uncertain_top_two_margin", -1))
    if not 0 <= margin <= 1:
        raise ValueError("clip.uncertain_top_two_margin must be between 0 and 1")
    for required in ("paths", "speciesnet", "birefnet", "clip"):
        if required not in values:
            raise ValueError(f"Configuration is missing 'part_a.{required}'")

    return PartAConfig(project_root, values)
