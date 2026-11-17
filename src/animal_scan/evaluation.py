from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.animal_scan.config import PartAConfig, load_config
from src.animal_scan.data import discover_images, load_labeled_examples
from src.animal_scan.pipeline import PartAPipeline
from src.utils.files import atomic_write_json
from src.utils.logger import configure_logging


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger("[Stage 1 evaluation]")


def _validate_examples(
    config: PartAConfig,
    examples: list[Any],
) -> tuple[list[Path], dict[Path, str]]:
    input_dir = config.paths["input_dir"].resolve()
    supported_images = {
        path.resolve()
        for path in discover_images(
            input_dir,
            tuple(config.values["image_extensions"]),
        )
    }
    labeled_paths = [example.image_path.resolve() for example in examples]
    duplicates = sorted(
        path for path, count in Counter(labeled_paths).items() if count > 1
    )
    if duplicates:
        raise ValueError(
            "Ground-truth CSV contains duplicate image paths: "
            + ", ".join(path.as_posix() for path in duplicates[:10])
        )

    folders_by_path: dict[Path, str] = {}
    for example, image_path in zip(examples, labeled_paths, strict=True):
        try:
            relative_path = image_path.relative_to(input_dir)
        except ValueError as error:
            raise ValueError(
                f"Ground-truth image is outside the configured input directory: {image_path}"
            ) from error
        if len(relative_path.parts) < 2:
            raise ValueError(
                f"Ground-truth image must be inside a subfolder of {input_dir}: {image_path}"
            )
        folder = relative_path.parts[0]
        if example.species != folder:
            raise ValueError(
                "Ground-truth species must match the image's top-level folder: "
                f"{relative_path.as_posix()} has species {example.species!r}, "
                f"but folder {folder!r}"
            )
        folders_by_path[image_path] = folder

    labeled_set = set(labeled_paths)
    missing = sorted(labeled_set - supported_images)
    unlabeled = sorted(supported_images - labeled_set)
    if missing or unlabeled:
        details = []
        if missing:
            details.append(
                "CSV images missing from the supported input inventory: "
                + ", ".join(path.as_posix() for path in missing[:10])
            )
        if unlabeled:
            details.append(
                "Supported input images missing ground-truth rows: "
                + ", ".join(path.as_posix() for path in unlabeled[:10])
            )
        raise ValueError("; ".join(details))
    if not examples:
        raise ValueError("Ground-truth CSV contains no labeled images")

    return labeled_paths, folders_by_path


def _metric(correct: int, total: int) -> dict[str, int | float | None]:
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else None,
    }


def evaluate_dataset(
    config: PartAConfig,
    workers: int | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Run Stage 1 over the configured labeled collection and save its accuracy report."""
    examples = load_labeled_examples(config)
    image_paths, folders_by_path = _validate_examples(config, examples)

    evaluation = config.values.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("Configuration must define part_a.evaluation as a mapping")
    effective_workers = int(
        workers if workers is not None else evaluation.get("workers", 1)
    )
    if effective_workers < 1:
        raise ValueError("Worker count must be at least 1")
    if device not in (None, "cpu", "cuda"):
        raise ValueError("Evaluation device must be 'cpu', 'cuda', or omitted")
    if device is not None:
        config.values["device"] = device
    requested_device = str(config.values.get("device", "auto"))
    if requested_device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise ValueError("GPU inference was requested, but CUDA is unavailable")

    report_path_value = evaluation.get("report_path")
    if not isinstance(report_path_value, str) or not report_path_value.strip():
        raise ValueError("Configuration must define part_a.evaluation.report_path")
    log_file_value = evaluation.get("log_file")
    if not isinstance(log_file_value, str) or not log_file_value.strip():
        raise ValueError("Configuration must define part_a.evaluation.log_file")
    report_path = Path(report_path_value)
    if not report_path.is_absolute():
        report_path = config.project_root / report_path
    config.values["paths"]["log_file"] = log_file_value
    config.values["concurrency"]["max_workers"] = effective_workers

    pipeline = PartAPipeline(config)
    if pipeline.device == "cuda":
        import torch

        device_name = torch.cuda.get_device_name()
        device_kind = "GPU"
    else:
        device_name = "CPU"
        device_kind = "CPU"
    device_info = {
        "requested": requested_device,
        "selected": pipeline.device,
        "kind": device_kind,
        "name": device_name,
        "models": ["SpeciesNet", "BiRefNet", "CLIP"],
    }
    LOGGER.info(
        "Inference device: %s (%s); workers: %d",
        device_name,
        pipeline.device,
        effective_workers,
    )
    pipeline.events.write(
        {
            "event": "evaluation_start",
            "image_count": len(image_paths),
            "workers": effective_workers,
            "inference_device": device_info,
        }
    )
    predictions = pipeline.process_paths(image_paths)
    if len(predictions) != len(examples):
        raise RuntimeError(
            f"Stage 1 returned {len(predictions)} predictions for {len(examples)} labeled images"
        )

    dataset_totals = {
        "total_images": len(examples),
        "species_correct": 0,
        "color_correct": 0,
    }
    folder_totals: dict[str, dict[str, int]] = {}
    status_counts: Counter[str] = Counter()
    for example, image_path, prediction in zip(
        examples, image_paths, predictions, strict=True
    ):
        folder = folders_by_path[image_path]
        totals = folder_totals.setdefault(
            folder,
            {"total_images": 0, "species_correct": 0, "color_correct": 0},
        )
        species_correct = prediction.get("species") == example.species
        color_correct = prediction.get("color") == example.color
        dataset_totals["species_correct"] += int(species_correct)
        dataset_totals["color_correct"] += int(color_correct)
        totals["total_images"] += 1
        totals["species_correct"] += int(species_correct)
        totals["color_correct"] += int(color_correct)
        status_counts[str(prediction.get("status", "unknown"))] += 1

    def summarize(totals: dict[str, int]) -> dict[str, Any]:
        count = totals["total_images"]
        return {
            "total_images": count,
            "species": _metric(totals["species_correct"], count),
            "color": _metric(totals["color_correct"], count),
        }

    report: dict[str, Any] = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ground_truth_csv": str(config.paths["labels_csv"]),
        "input_dir": str(config.paths["input_dir"]),
        "workers": effective_workers,
        "inference_device": device_info,
        "dataset": summarize(dataset_totals),
        "by_subfolder": {
            folder: summarize(totals)
            for folder, totals in sorted(folder_totals.items())
        },
        "prediction_status_counts": dict(sorted(status_counts.items())),
        "report_path": str(report_path),
    }

    atomic_write_json(report_path, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate Stage 1 species and color accuracy on labeled images."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs.yaml",
        help="Shared YAML configuration.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Override part_a.evaluation.workers for this evaluation.",
    )
    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument(
        "--cpu",
        dest="device",
        action="store_const",
        const="cpu",
        help="Run Stage 1 inference on the CPU.",
    )
    device_group.add_argument(
        "--gpu",
        dest="device",
        action="store_const",
        const="cuda",
        help="Run Stage 1 inference on the CUDA GPU.",
    )
    parser.set_defaults(device=None)
    arguments = parser.parse_args(argv)
    if arguments.workers is not None and arguments.workers < 1:
        parser.error("--workers must be at least 1")

    configure_logging()
    try:
        config = load_config(arguments.config)
        report = evaluate_dataset(
            config,
            workers=arguments.workers,
            device=arguments.device,
        )
    except (OSError, ValueError, RuntimeError, KeyError, yaml.YAMLError) as error:
        print(f"Stage 1 evaluation failed: {error}", file=sys.stderr)
        return 2

    print(
        f"Evaluated {report['dataset']['total_images']} images. "
        f"Species accuracy: {report['dataset']['species']['accuracy']:.2%}; "
        f"color accuracy: {report['dataset']['color']['accuracy']:.2%}."
    )
    print(f"Report: {report['report_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
