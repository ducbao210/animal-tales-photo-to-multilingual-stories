"""Shared, versioned value objects used at the A--D stage boundaries."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "1.0"
LOCALES = ("en", "fr-FR", "es-ES", "vi-VN")
TARGET_LOCALES = LOCALES[1:]
SECTIONS = ("beginning", "problem", "solution", "happy_ending")


class ContractError(ValueError):
    """Raised when persisted or inter-stage data violates its contract."""


class StageError(RuntimeError):
    def __init__(self, step: str, category: str, retryable: bool, message: str) -> None:
        super().__init__(message)
        self.step = step
        self.category = category
        self.retryable = retryable
        self.message = message

    def to_dict(self, *, attempt: int = 1) -> dict[str, Any]:
        from datetime import datetime, timezone

        return {
            "stage": self.step,
            "category": self.category,
            "message": self.message,
            "retryable": self.retryable,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "attempt": attempt,
        }


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be an object")
    return value


def _text(value: Any, label: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ContractError(f"{label} must be a non-empty string")
    return value


def _locales(value: Any, label: str, expected: tuple[str, ...] = LOCALES) -> dict[str, str]:
    mapping = _mapping(value, label)
    if set(mapping) != set(expected):
        raise ContractError(f"{label} must define exactly {', '.join(expected)}")
    return {locale: _text(mapping[locale], f"{label}.{locale}") for locale in expected}


@dataclass(frozen=True)
class ScanResult:
    image_id: str
    input_path: str
    species: str
    color: str | None
    status: str
    species_uncertain: bool
    color_uncertain: bool
    errors: tuple[dict[str, Any], ...] = ()
    raw: dict[str, Any] = field(default_factory=dict, compare=False)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.pop("raw", None)
        result["schema_version"] = SCHEMA_VERSION
        result["errors"] = list(self.errors)
        return result


def validate_scan(
    raw: dict[str, Any], allowed_species: set[str], allowed_colors: set[str],
    unknown_species: str = "unknown",
) -> ScanResult:
    value = _mapping(raw, "scan")
    if value.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
        raise ContractError("Unsupported scan schema_version")
    species = _text(value.get("species"), "scan.species")
    color = value.get("color")
    if species not in (*allowed_species, unknown_species):
        raise ContractError(f"Unsupported species label: {species}")
    if color is not None and color not in allowed_colors:
        raise ContractError(f"Unsupported color label: {color}")
    status = _text(value.get("status"), "scan.status")
    if status not in {"complete", "partial", "failed"}:
        raise ContractError(f"Unsupported Stage 1 status: {status}")
    errors = value.get("errors", [])
    if not isinstance(errors, list) or not all(isinstance(item, dict) for item in errors):
        raise ContractError("scan.errors must be a list of objects")
    for flag in ("species_uncertain", "color_uncertain"):
        if not isinstance(value.get(flag), bool):
            raise ContractError(f"scan.{flag} must be a boolean")
    return ScanResult(
        image_id=_text(value.get("image_id"), "scan.image_id"),
        input_path=_text(value.get("input_path"), "scan.input_path"),
        species=species,
        color=color,
        status=status,
        species_uncertain=value["species_uncertain"],
        color_uncertain=value["color_uncertain"],
        errors=tuple(errors),
        raw=dict(value),
    )


@dataclass(frozen=True)
class MainProfile:
    species: str
    name: str
    gender: str
    pronouns_en: str
    slogan_en: str
    skill: str
    visual_detail: str
    names: dict[str, str]
    slogans: dict[str, str]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MainProfile":
        value = _mapping(value, "main profile")
        return cls(
            species=_text(value.get("species"), "profile.species"),
            name=_text(value.get("name"), "profile.name"),
            gender=_text(value.get("gender"), "profile.gender"),
            pronouns_en=_text(value.get("pronouns_en"), "profile.pronouns_en"),
            slogan_en=_text(value.get("slogan_en"), "profile.slogan_en"),
            skill=_text(value.get("skill"), "profile.skill"),
            visual_detail=_text(value.get("visual_detail"), "profile.visual_detail"),
            names=_locales(value.get("names"), "profile.names"),
            slogans=_locales(value.get("slogans"), "profile.slogans"),
        )


@dataclass(frozen=True)
class StoryWorld:
    location: dict[str, Any]
    main: dict[str, MainProfile]
    punctuation: dict[str, str]
    pronouns: dict[str, Any]

    @classmethod
    def from_dict(cls, value: dict[str, Any], species: set[str]) -> "StoryWorld":
        value = _mapping(value, "story world")
        location = _mapping(value.get("location"), "location")
        _locales(location.get("names"), "location.names")
        areas = location.get("areas")
        if not isinstance(areas, list) or len(areas) not in (4, 5):
            raise ContractError("location.areas must contain four or five areas")
        area_ids: set[str] = set()
        for area in areas:
            area = _mapping(area, "area")
            area_id = _text(area.get("id"), "area.id")
            if area_id in area_ids:
                raise ContractError(f"Duplicate area id: {area_id}")
            area_ids.add(area_id)
            _locales(area.get("names"), f"area.{area_id}.names")
            _text(area.get("description"), f"area.{area_id}.description")
        main_data = _mapping(value.get("main"), "main")
        if set(main_data) != species:
            raise ContractError("story world must define every configured species exactly once")
        main = {key: MainProfile.from_dict({"species": key, **main_data[key]}) for key in species}
        punctuation = _mapping(value.get("punctuation"), "punctuation")
        if set(punctuation) != set(TARGET_LOCALES):
            raise ContractError("punctuation must define fr-FR, es-ES, and vi-VN")
        return cls(location, main, punctuation, _mapping(value.get("pronouns"), "pronouns"))

    def areas_by_id(self) -> dict[str, dict[str, Any]]:
        return {area["id"]: area for area in self.location["areas"]}


@dataclass(frozen=True)
class CharacterResult:
    schema_version: str
    image_id: str
    species: str
    color: str
    profile_id: str
    name: str
    image_path: str
    model: str
    backend: str
    parameters: dict[str, Any]
    prompt_sha256: str
    request_id: str | None
    width: int
    height: int
    mime_type: str
    status: str = "complete"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StoryPage:
    page: int
    section: str
    area_id: str
    text: str
    illustration_prompt: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StoryResult:
    schema_version: str
    image_id: str
    title: str
    language: str
    character_id: str
    species: str
    color: str
    support_id: str
    support_color: str
    location_id: str
    pages: tuple[StoryPage, ...]
    validation: dict[str, Any]
    model: str
    prompt_sha256: str

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["pages"] = [page.to_dict() for page in self.pages]
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "StoryResult":
        value = _mapping(value, "story")
        if value.get("schema_version") != SCHEMA_VERSION:
            raise ContractError("Unsupported story schema_version")
        pages_data = value.get("pages")
        if not isinstance(pages_data, list):
            raise ContractError("story.pages must be a list")
        pages = []
        for page in pages_data:
            page = _mapping(page, "story page")
            pages.append(StoryPage(
                page=page.get("page"), section=_text(page.get("section"), "page.section"),
                area_id=_text(page.get("area_id"), "page.area_id"), text=_text(page.get("text"), "page.text"),
                illustration_prompt=page.get("illustration_prompt"),
            ))
        return cls(
            schema_version=SCHEMA_VERSION,
            image_id=_text(value.get("image_id"), "story.image_id"),
            title=_text(value.get("title"), "story.title"),
            language=_text(value.get("language"), "story.language"),
            character_id=_text(value.get("character_id"), "story.character_id"),
            species=_text(value.get("species"), "story.species"), color=_text(value.get("color"), "story.color"),
            support_id=_text(value.get("support_id"), "story.support_id"),
            support_color=_text(value.get("support_color"), "story.support_color"),
            location_id=_text(value.get("location_id"), "story.location_id"),
            pages=tuple(pages), validation=_mapping(value.get("validation"), "story.validation"),
            model=_text(value.get("model"), "story.model"), prompt_sha256=_text(value.get("prompt_sha256"), "story.prompt_sha256"),
        )


@dataclass(frozen=True)
class Translation:
    locale: str
    pages: tuple[dict[str, Any], ...]
    validation: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"locale": self.locale, "pages": list(self.pages), "validation": self.validation}


@dataclass(frozen=True)
class TranslationBundle:
    schema_version: str
    image_id: str
    source_story_sha256: str
    translations: dict[str, Translation]
    validation: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version, "image_id": self.image_id,
            "source_story_sha256": self.source_story_sha256,
            "translations": {key: value.to_dict() for key, value in self.translations.items()},
            "validation": self.validation,
        }


def ensure_under(path: str | Path, root: str | Path) -> Path:
    candidate = Path(path).resolve()
    base = Path(root).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as error:
        raise ContractError("Artifact path must stay within the configured output root") from error
    return candidate
