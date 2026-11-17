import argparse
import os
from pathlib import Path
import random
import shutil
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.animal_scan.config import load_config_document, resolve_path

DEFAULT_CONFIG = PROJECT_ROOT / "configs.yaml"


def images(folder: Path, image_extensions: set[str]) -> list[Path]:
    return sorted(
        (
            p
            for p in folder.iterdir()
            if p.is_file() and p.suffix.casefold() in image_extensions
        ),
        key=lambda p: p.name,
    )


def find_dataset(path: Path, species: list[str]) -> Path | None:
    if not path.exists():
        return None

    return next(
        (
            p
            for p in path.rglob("*")
            if p.is_dir() and all((p / name).is_dir() for name in species)
        ),
        None,
    )


def clean_and_rename(
    folder: Path,
    remove_ids: set[int] | set[str],
    image_extensions: set[str],
    unknown_species: str,
) -> None:
    # Delete unwanted images
    for image in images(folder, image_extensions):
        if folder.name == unknown_species:
            should_remove = image.stem in remove_ids
        else:
            should_remove = int(image.stem) in remove_ids

        if should_remove:
            image.unlink()

    # Unknown images retain their original filenames
    if folder.name == unknown_species:
        return

    # Sort remaining images by original numeric ID
    remaining = sorted(
        images(folder, image_extensions),
        key=lambda p: int(p.stem),
    )

    # Temporary rename to prevent filename conflicts
    temp_files = []

    for index, image in enumerate(remaining, 1):
        temp = folder / f"__temp_{index}{image.suffix.lower()}"
        image.rename(temp)
        temp_files.append(temp)

    # Rename sequentially from 1 to N
    for index, image in enumerate(temp_files, 1):
        image.rename(folder / f"{index}{image.suffix.lower()}")


def _load_settings(config_path: Path) -> tuple[Path, dict[str, Any]]:
    project_root, document = load_config_document(config_path)
    part_a = document.get("part_a")
    dataset = document.get("dataset")
    if not isinstance(part_a, dict) or not isinstance(dataset, dict):
        raise ValueError("Configuration must contain 'part_a' and 'dataset' mappings")

    taxonomy = part_a.get("taxonomy")
    paths = part_a.get("paths")
    if not isinstance(taxonomy, dict) or not isinstance(paths, dict):
        raise ValueError("Configuration must define part_a.taxonomy and part_a.paths")
    species = taxonomy.get("species")
    unknown_species = taxonomy.get("unknown_species")
    unknown_source_classes = dataset.get("unknown_source_classes")
    remove_images = dataset.get("remove_images")
    image_extensions = part_a.get("image_extensions")
    if (
        not isinstance(species, list)
        or not species
        or not all(isinstance(label, str) and label for label in species)
        or not isinstance(unknown_species, str)
        or not unknown_species
        or unknown_species in species
        or not isinstance(unknown_source_classes, list)
        or not all(isinstance(label, str) and label for label in unknown_source_classes)
        or len(set(unknown_source_classes)) != len(unknown_source_classes)
        or set(species) & set(unknown_source_classes)
        or not isinstance(remove_images, dict)
        or not isinstance(image_extensions, list)
        or not image_extensions
        or not all(
            isinstance(extension, str) and extension.startswith(".")
            for extension in image_extensions
        )
    ):
        raise ValueError("Dataset and part_a taxonomy/image settings are invalid")
    if "input_dir" not in paths or not isinstance(paths["input_dir"], str):
        raise ValueError("Configuration must define part_a.paths.input_dir")
    if not isinstance(dataset.get("source"), str) or not dataset["source"]:
        raise ValueError("Configuration must define dataset.source")
    if not isinstance(dataset.get("cache_dir"), str) or not dataset["cache_dir"]:
        raise ValueError("Configuration must define dataset.cache_dir")
    for label, remove_ids in remove_images.items():
        if label not in (*species, unknown_species) or not isinstance(remove_ids, list):
            raise ValueError(f"Invalid dataset.remove_images entry for {label!r}")
        expected_type = str if label == unknown_species else int
        if not all(type(remove_id) is expected_type for remove_id in remove_ids):
            raise ValueError(f"Dataset cleanup IDs for {label!r} have invalid values")

    settings = {
        "source": dataset["source"],
        "cache_dir": resolve_path(project_root, dataset["cache_dir"]),
        "output": resolve_path(project_root, paths["input_dir"]),
        "species": species,
        "unknown_species": unknown_species,
        "unknown_source_classes": unknown_source_classes,
        "remove_images": remove_images,
        "image_extensions": {extension.casefold() for extension in image_extensions},
    }
    return project_root, settings


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Create the configured animal dataset."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    arguments = parser.parse_args(argv)
    _project_root, settings = _load_settings(arguments.config)
    species = settings["species"]
    unknown_species = settings["unknown_species"]
    unknown_source_classes = settings["unknown_source_classes"]
    remove_images = settings["remove_images"]
    image_extensions = settings["image_extensions"]
    cache = settings["cache_dir"]
    output = settings["output"]

    os.environ["KAGGLEHUB_CACHE"] = str(cache)
    import kagglehub

    cache_dataset = cache / "datasets" / Path(settings["source"])
    dataset_path = find_dataset(cache_dataset, species)

    # Download if not cached
    if dataset_path is None:
        try:
            dataset_path = Path(kagglehub.dataset_download(settings["source"]))
        except PermissionError:
            dataset_path = find_dataset(cache_dataset, species)
            if dataset_path is None:
                raise

    # Locate class folders
    folders = {p.name.casefold(): p for p in dataset_path.rglob("*") if p.is_dir()}

    # Validate source folders
    for label in (*species, *unknown_source_classes):
        if label not in folders:
            raise FileNotFoundError(f"Dataset is missing class folder: {label}")

    # Recreate output directory
    if output.exists():
        shutil.rmtree(output)

    output.mkdir(parents=True, exist_ok=True)

    for label in (*species, unknown_species):
        (output / label).mkdir(exist_ok=True)

    # Copy main classes
    for label in species:
        source = folders[label]

        for index, image in enumerate(images(source, image_extensions), 1):
            shutil.copy2(image, output / label / f"{index}{image.suffix.lower()}")

    # Sample unknown classes
    rng = random.Random(42)
    unknown = output / unknown_species

    for label in unknown_source_classes:
        source_images = images(folders[label], image_extensions)

        selected = rng.sample(
            source_images,
            min(15, len(source_images)),
        )

        for index, image in enumerate(selected, 1):
            shutil.copy2(image, unknown / f"{label}_{index}{image.suffix.lower()}")

    # Remove unwanted images and renumber
    for label, remove_ids in remove_images.items():
        if label not in (*species, unknown_species):
            raise ValueError(f"Unsupported dataset cleanup class: {label}")
        clean_and_rename(
            output / label,
            set(remove_ids),
            image_extensions,
            unknown_species,
        )

    # Validate and print summary
    print(f"\nCreated dataset at {output.resolve()}")
    print("-" * 40)

    total = 0

    for folder in sorted(output.iterdir()):
        if not folder.is_dir():
            continue

        count = len(images(folder, image_extensions))
        total += count

        if folder.name in species:
            ids = sorted(int(p.stem) for p in images(folder, image_extensions))

            assert ids == list(
                range(1, count + 1)
            ), f"Invalid numbering in {folder.name}"

        print(f"{folder.name}: {count} images")

    print("-" * 40)
    print(f"Total: {total} images")


if __name__ == "__main__":
    main()
