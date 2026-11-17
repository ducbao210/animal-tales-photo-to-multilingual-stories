from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from torch.utils.data import DataLoader, Dataset

from .config import PartAConfig


@dataclass(frozen=True)
class LabeledExample:
    image_path: Path
    species: str
    color: str


class LabeledImageDataset(Dataset[dict[str, str]]):
    """The current labeled collection; no train/dev/test split is created."""

    def __init__(self, examples: list[LabeledExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, str]:
        example = self.examples[index]
        return {
            "image_path": str(example.image_path),
            "species": example.species,
            "color": example.color,
        }


def load_labeled_examples(config: PartAConfig) -> list[LabeledExample]:
    csv_path = config.paths["labels_csv"]
    examples: list[LabeledExample] = []
    with csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["relative_path", "species", "color"]:
            raise ValueError(
                f"{csv_path} must have columns relative_path,species,color"
            )
        for row_number, row in enumerate(reader, start=2):
            relative_path = row["relative_path"]
            image_path = config.project_root / relative_path
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Labeled image on row {row_number} is missing: {relative_path}"
                )
            if row["species"] not in (
                *config.taxonomy["species"],
                config.taxonomy["unknown_species"],
            ):
                raise ValueError(f"Unsupported species label on row {row_number}")
            if row["color"] not in config.taxonomy["colors"]:
                raise ValueError(f"Unsupported color label on row {row_number}")
            examples.append(LabeledExample(image_path, row["species"], row["color"]))
    return examples


def create_test_loader(
    config: PartAConfig, batch_size: int = 32
) -> DataLoader[dict[str, list[str]]]:
    """Load all available labels for descriptive evaluation without a held-out split."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    return DataLoader(
        LabeledImageDataset(load_labeled_examples(config)),
        batch_size=batch_size,
        shuffle=False,
    )


def discover_images(
    input_dir: Path, extensions: tuple[str, ...], limit: int | None = None
) -> Iterator[Path]:
    normalized_extensions = {extension.casefold() for extension in extensions}
    images = sorted(
        (
            path
            for path in input_dir.rglob("*")
            if path.is_file() and path.suffix.casefold() in normalized_extensions
        ),
        key=lambda path: path.as_posix().casefold(),
    )
    yield from images if limit is None else images[:limit]
