from __future__ import annotations

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from src.utils.files import atomic_write_json
from src.utils.logger import JsonlEventLogger

from .config import PartAConfig
from .models import (
    BiRefNetSegmenter,
    OpenAIClipColorClassifier,
    SegmentPrediction,
    SpeciesNetRunner,
    SpeciesPrediction,
    crop_species_box,
    map_species_label,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PartAPipeline:
    def __init__(
        self,
        config: PartAConfig,
        speciesnet: Any | None = None,
        birefnet: Any | None = None,
        clip_classifier: Any | None = None,
        event_logger: JsonlEventLogger | None = None,
        refresh_speciesnet: bool = False,
    ) -> None:
        self.config = config
        settings = config.values
        device = settings["device"]
        if device == "auto":
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.speciesnet = speciesnet or SpeciesNetRunner(
            {**settings["speciesnet"], "device": device},
            force_model_download=refresh_speciesnet,
        )
        self.birefnet = birefnet or BiRefNetSegmenter(settings["birefnet"], device)
        self.clip_classifier = clip_classifier or OpenAIClipColorClassifier(
            settings["clip"], device
        )
        self.paths = config.paths
        self.events = event_logger or JsonlEventLogger(self.paths["log_file"])
        self.aliases = config.taxonomy["species_aliases"]

    def process_paths(self, image_paths: list[Path]) -> list[dict[str, Any]]:
        max_workers = int(self.config.values["concurrency"]["max_workers"])
        with ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="part-a-photo"
        ) as executor:
            return list(executor.map(self.process_one, image_paths))

    def process_one(self, image_path: Path) -> dict[str, Any]:
        image_id = self._image_id(image_path)
        input_path = self._display_path(image_path)
        errors: list[dict[str, str]] = []
        notes: list[str] = []
        started = time.perf_counter()
        try:
            self.events.write(
                {
                    "timestamp": _now(),
                    "event": "step_start",
                    "image_id": image_id,
                    "input_path": input_path,
                    "step": "input",
                    "attempt": 1,
                    "status": "running",
                }
            )
            image = Image.open(image_path).convert("RGB")
        except (OSError, ValueError) as error:
            self.events.write(
                {
                    "timestamp": _now(),
                    "event": "step_end",
                    "image_id": image_id,
                    "input_path": input_path,
                    "step": "input",
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                    "status": "failed",
                    "error": str(error),
                }
            )
            result = self._empty_result(image_id, input_path, str(error))
            self._save_result(image_id, result)
            return result
        self.events.write(
            {
                "timestamp": _now(),
                "event": "step_end",
                "image_id": image_id,
                "input_path": input_path,
                "step": "input",
                "attempt": 1,
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                "status": "succeeded",
            }
        )

        species_prediction: SpeciesPrediction | None = None
        species, species_error = "unknown", None
        success, value, error = self._run_step(
            image_id,
            input_path,
            "speciesnet",
            lambda: self.speciesnet.predict(image_path),
        )
        if success:
            species_prediction = value
            species, species_error = map_species_label(
                value.label,
                self.aliases,
                self.config.taxonomy["unknown_species"],
            )
            if species_error:
                notes.append(species_error)
        else:
            errors.append(
                {"step": "speciesnet", "message": error or "SpeciesNet failed"}
            )
            notes.append("SpeciesNet failed; species is unknown.")

        segment: SegmentPrediction | None = None
        success, value, error = self._run_step(
            image_id, input_path, "birefnet", lambda: self.birefnet.segment(image)
        )
        if success and isinstance(value, SegmentPrediction):
            segment = value
        elif success:
            success = False
            error = "BiRefNet returned an invalid foreground result."
        if not success:
            errors.append({"step": "birefnet", "message": error or "BiRefNet failed"})
            notes.append(
                "BiRefNet foreground extraction failed; color uses the best available "
                "fallback evidence and is marked uncertain."
            )

        color_input = (
            segment.color_image
            if segment
            else crop_species_box(
                image, species_prediction.bounding_box if species_prediction else None
            )
        )
        foreground_status = "birefnet" if segment else "fallback"
        saved, foreground_ref, save_error = self._run_step(
            image_id,
            input_path,
            "foreground_save",
            lambda: self._save_foreground(
                image_id, segment.foreground_rgba if segment else color_input
            ),
        )
        if not saved:
            foreground_ref = None
            errors.append(
                {"step": "foreground_save", "message": save_error or "Save failed"}
            )
            notes.append("The foreground/fallback evidence could not be saved.")
        if not segment and species_prediction and species_prediction.bounding_box:
            notes.append(
                "Fallback color evidence uses the SpeciesNet animal bounding box."
            )
        elif not segment:
            notes.append("Fallback color evidence uses the original image.")

        color: str | None = None
        color_scores: dict[str, float] | None = None
        color_uncertain = True
        color_uncertainty_reason: str | None = None
        if color_input is None:
            errors.append(
                {"step": "color", "message": "No usable image evidence was available."}
            )
            color_uncertainty_reason = "No usable image evidence was available."
        else:
            success, value, error = self._run_step(
                image_id,
                input_path,
                "clip_color",
                lambda: self._validated_color_scores(
                    self.clip_classifier.predict(color_input)
                ),
            )
            if success:
                color_scores = value
                color = max(color_scores, key=color_scores.__getitem__)
                ordered_scores = sorted(color_scores.values(), reverse=True)
                margin = ordered_scores[0] - ordered_scores[1]
                color_uncertain = margin <= float(
                    self.config.values["clip"]["uncertain_top_two_margin"]
                )
                if color_uncertain:
                    color_uncertainty_reason = (
                        "CLIP's configured top-two softmax-score margin is at or below "
                        "the diagnostic threshold."
                    )
                if not segment:
                    color_uncertain = True
                    color_uncertainty_reason = "Foreground extraction failed; CLIP used fallback image evidence."
            else:
                errors.append({"step": "clip_color", "message": error or "CLIP failed"})
                color_uncertainty_reason = "CLIP color classification failed."

        result_status = "complete"
        if errors and color is None:
            result_status = "failed"
        elif errors or color_uncertain or species == "unknown":
            result_status = "partial"
        result = {
            "schema_version": "1.0",
            "image_id": image_id,
            "input_path": input_path,
            "species": species,
            "species_confidence": (
                species_prediction.confidence if species_prediction else None
            ),
            "species_uncertain": species == "unknown",
            "color": color,
            "color_confidence": (
                color_scores.get(color) if color and color_scores else None
            ),
            "color_scores": color_scores,
            "color_uncertain": color_uncertain,
            "color_uncertainty_reason": color_uncertainty_reason,
            "foreground_status": foreground_status,
            "foreground_path": foreground_ref,
            "mapping_notes": [species_error] if species_error else [],
            "processing_notes": notes,
            "models": {
                "species": self.config.values["speciesnet"]["model_id"],
                "foreground": self.config.values["birefnet"]["model_id"],
                "color": self.config.values["clip"]["model_id"],
            },
            "timestamp": _now(),
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            "status": result_status,
            "errors": errors,
        }
        self._save_result(image_id, result)
        return result

    def _run_step(
        self,
        image_id: str,
        input_path: str,
        step: str,
        operation: Callable[[], Any],
    ) -> tuple[bool, Any, str | None]:
        retries = self.config.values["retries"]
        attempts = int(retries["max_attempts"])
        delay = float(retries["retry_delay_seconds"])
        for attempt in range(1, attempts + 1):
            started = time.perf_counter()
            self.events.write(
                {
                    "timestamp": _now(),
                    "event": "step_start",
                    "image_id": image_id,
                    "input_path": input_path,
                    "step": step,
                    "attempt": attempt,
                    "status": "running",
                }
            )
            try:
                value = operation()
            except Exception as error:
                message = f"{type(error).__name__}: {error}"
                duration_ms = round((time.perf_counter() - started) * 1000, 2)
                if attempt < attempts:
                    self.events.write(
                        {
                            "timestamp": _now(),
                            "event": "step_retry",
                            "image_id": image_id,
                            "input_path": input_path,
                            "step": step,
                            "attempt": attempt,
                            "duration_ms": duration_ms,
                            "status": "retrying",
                            "error": message,
                        }
                    )
                    if delay:
                        time.sleep(delay)
                    continue
                self.events.write(
                    {
                        "timestamp": _now(),
                        "event": "step_end",
                        "image_id": image_id,
                        "input_path": input_path,
                        "step": step,
                        "attempt": attempt,
                        "duration_ms": duration_ms,
                        "status": "failed",
                        "error": message,
                    }
                )
                return False, None, message
            self.events.write(
                {
                    "timestamp": _now(),
                    "event": "step_end",
                    "image_id": image_id,
                    "input_path": input_path,
                    "step": step,
                    "attempt": attempt,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                    "status": "succeeded",
                }
            )
            return True, value, None
        raise RuntimeError("Retry loop exited without a result.")

    def _validated_color_scores(self, scores: Any) -> dict[str, float]:
        colors = self.config.taxonomy["colors"]
        if not isinstance(scores, dict) or set(scores) != set(colors):
            raise ValueError("CLIP must return one score for each configured color.")
        validated = {color: float(scores[color]) for color in colors}
        if any(not 0 <= score <= 1 for score in validated.values()):
            raise ValueError("CLIP softmax scores must be between zero and one.")
        return validated

    def _image_id(self, image_path: Path) -> str:
        try:
            display_name = (
                image_path.resolve()
                .relative_to(self.paths["input_dir"].resolve())
                .as_posix()
            )
        except ValueError:
            display_name = image_path.resolve().as_posix()
        digest = hashlib.sha256(display_name.encode("utf-8")).hexdigest()[:12]
        return f"{Path(display_name).stem}-{digest}"

    def _display_path(self, image_path: Path) -> str:
        try:
            return image_path.resolve().relative_to(self.config.project_root).as_posix()
        except ValueError:
            return image_path.resolve().as_posix()

    def _save_foreground(self, image_id: str, image: Image.Image | None) -> str | None:
        if image is None:
            return None
        directory = self.paths["foreground_dir"]
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f"{image_id}.png"
        image.save(destination, format="PNG")
        return self._display_path(destination)

    def _save_result(self, image_id: str, result: dict[str, Any]) -> None:
        directory = self.paths["results_dir"]
        destination = directory / f"{image_id}.json"
        atomic_write_json(destination, result)

    def _empty_result(
        self, image_id: str, input_path: str, error: str
    ) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "image_id": image_id,
            "input_path": input_path,
            "species": "unknown",
            "species_confidence": None,
            "species_uncertain": True,
            "color": None,
            "color_confidence": None,
            "color_scores": None,
            "color_uncertain": True,
            "color_uncertainty_reason": "Input image could not be opened.",
            "foreground_status": "unavailable",
            "foreground_path": None,
            "mapping_notes": [],
            "processing_notes": [],
            "models": {
                "species": self.config.values["speciesnet"]["model_id"],
                "foreground": self.config.values["birefnet"]["model_id"],
                "color": self.config.values["clip"]["model_id"],
            },
            "timestamp": _now(),
            "duration_ms": None,
            "status": "failed",
            "errors": [{"step": "input", "message": error}],
        }
