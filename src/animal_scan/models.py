from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image


@dataclass(frozen=True)
class SpeciesPrediction:
    label: str | None
    confidence: float | None
    bounding_box: tuple[float, float, float, float] | None


@dataclass(frozen=True)
class SegmentPrediction:
    foreground_rgba: Image.Image
    color_image: Image.Image


def normalize_label(value: str) -> str:
    return " ".join(value.casefold().replace("_", " ").replace("-", " ").split())


def map_species_label(
    label: str | None,
    aliases: dict[str, list[str]],
    unknown_species: str = "unknown",
) -> tuple[str, str | None]:
    if not label:
        return unknown_species, "SpeciesNet returned no classification label."

    raw_label = label.strip()
    if ": " in raw_label:
        prefix, remainder = raw_label.split(": ", 1)
        if prefix.isdigit():
            raw_label = remainder

    candidates = {normalize_label(raw_label)}
    if ";" in raw_label:
        fields = [field.strip() for field in raw_label.split(";")]
        # SpeciesNet classifier labels are UUID; class; order; family; genus;
        # species; common name. Match aliases against complete fields only.
        candidates.update(normalize_label(field) for field in fields[1:] if field)
        if len(fields) > 5 and fields[4] and fields[5]:
            candidates.add(normalize_label(f"{fields[4]} {fields[5]}"))

    for species, accepted_labels in aliases.items():
        if candidates.intersection(normalize_label(item) for item in accepted_labels):
            return species, None
    return unknown_species, f"SpeciesNet label {label!r} is not an exact configured alias."


def _classification(row: dict[str, Any]) -> tuple[str | None, float | None]:
    for key in (
        "processed_classification",
        "raw_classifications",
        "classifications",
        "classification",
        "prediction",
    ):
        value = row.get(key)
        if isinstance(value, list):
            value = value[0] if value else None
        if isinstance(value, dict):
            classes = value.get("classes", value.get("class"))
            scores = value.get("scores", value.get("score"))
            if isinstance(classes, list):
                label = classes[0] if classes else None
                confidence = scores[0] if isinstance(scores, list) and scores else None
                if label is not None:
                    score = float(confidence) if confidence is not None else None
                    if score is not None and (
                        not math.isfinite(score) or not 0 <= score <= 1
                    ):
                        raise ValueError(
                            "SpeciesNet returned an invalid classification score."
                        )
                    return str(label), score
            elif isinstance(classes, str) and classes:
                score = float(scores) if scores is not None else None
                if score is not None and (
                    not math.isfinite(score) or not 0 <= score <= 1
                ):
                    raise ValueError(
                        "SpeciesNet returned an invalid classification score."
                    )
                return classes, score
        elif isinstance(value, str) and value:
            return value, None
    return None, None


def _animal_detection(row: dict[str, Any]) -> dict[str, Any] | None:
    detections = row.get("detections", [])
    if isinstance(detections, dict):
        detections = [detections]
    if not isinstance(detections, list):
        return None
    animals = [
        detection
        for detection in detections
        if isinstance(detection, dict)
        and str(detection.get("category", "")).casefold()
        in {"1", "animal", "animal_detection"}
    ]
    if not animals:
        return None
    return max(animals, key=lambda item: float(item.get("conf", 0.0)))


def _parse_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        return None
    try:
        bbox = tuple(float(part) for part in value)
    except (TypeError, ValueError):
        return None
    if not all(0 <= part <= 1 for part in bbox) or bbox[2] <= 0 or bbox[3] <= 0:
        return None
    return cast(tuple[float, float, float, float], bbox)


def parse_speciesnet_output(document: dict[str, Any]) -> SpeciesPrediction:
    predictions = document.get("predictions")
    row = predictions[0] if isinstance(predictions, list) and predictions else document
    if not isinstance(row, dict):
        return SpeciesPrediction(None, None, None)
    detection = _animal_detection(row)
    if detection is None:
        return SpeciesPrediction(None, None, None)
    label, confidence = _classification(detection)
    if label is None:
        label, confidence = _classification(row)
    return SpeciesPrediction(label, confidence, _parse_bbox(detection.get("bbox")))


def _crop_box(
    bbox: tuple[float, float, float, float], size: tuple[int, int]
) -> tuple[int, int, int, int]:
    width, height = size
    x, y, box_width, box_height = bbox
    left = max(0, min(width - 1, round(x * width)))
    top = max(0, min(height - 1, round(y * height)))
    right = max(left + 1, min(width, round((x + box_width) * width)))
    bottom = max(top + 1, min(height, round((y + box_height) * height)))
    return left, top, right, bottom


class SpeciesNetRunner:
    def __init__(
        self, settings: dict[str, Any], force_model_download: bool = False
    ) -> None:
        self.settings = settings
        self.force_model_download = force_model_download
        self._refresh_lock = threading.Lock()

    def predict(self, image_path: Path) -> SpeciesPrediction:
        if not self.force_model_download:
            return self._predict(image_path, force_model_download=False)
        with self._refresh_lock:
            force_download = self.force_model_download
            prediction = self._predict(image_path, force_model_download=force_download)
            self.force_model_download = False
            return prediction

    def _predict(
        self, image_path: Path, force_model_download: bool
    ) -> SpeciesPrediction:
        with tempfile.TemporaryDirectory(prefix="part-a-speciesnet-") as temporary:
            folder = Path(temporary) / "image"
            folder.mkdir()
            copied_image = folder / image_path.name
            shutil.copy2(image_path, copied_image)
            output_file = Path(temporary) / "predictions.json"
            command = list(self.settings["command"])
            if command and command[0].casefold() == "python":
                command[0] = sys.executable
            command.extend(["--model", self.settings["model_id"]])
            command.extend(
                [
                    "--folders",
                    str(folder),
                    "--predictions_json",
                    str(output_file),
                ]
            )
            if force_model_download:
                command.append("--force_model_download")
            environment = None
            if self.settings.get("device") == "cpu":
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = ""
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=int(self.settings["timeout_seconds"]),
                check=False,
                env=environment,
            )
            if completed.returncode:
                detail = completed.stderr.strip() or completed.stdout.strip()
                raise RuntimeError(
                    f"SpeciesNet exited with code {completed.returncode}: {detail[-2000:]}"
                )
            if not output_file.is_file():
                raise RuntimeError(
                    "SpeciesNet completed without writing its JSON output."
                )
            with output_file.open(encoding="utf-8") as stream:
                document = json.load(stream)
        return parse_speciesnet_output(document)


class BiRefNetSegmenter:
    def __init__(self, settings: dict[str, Any], device: str) -> None:
        self.settings = settings
        self.device = device
        self.model: Any = None

    def _load(self) -> None:
        if self.model is None:
            from transformers import AutoModelForImageSegmentation

            self.model = AutoModelForImageSegmentation.from_pretrained(
                self.settings["model_id"],
                trust_remote_code=bool(self.settings["trust_remote_code"]),
            ).to(self.device)
            self.model.eval()

    def segment(self, image: Image.Image) -> SegmentPrediction:
        self._load()
        size = int(self.settings["image_size"])
        resized = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        pixels = torch.from_numpy(
            np.array(resized, dtype=np.float32, copy=True) / 255.0
        )
        pixels = pixels.permute(2, 0, 1)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        batch = ((pixels - mean) / std).unsqueeze(0)
        parameters = getattr(self.model, "parameters", None)
        model_parameter = next(parameters(), None) if callable(parameters) else None
        model_dtype = (
            model_parameter.dtype if model_parameter is not None else batch.dtype
        )
        batch = batch.to(device=self.device, dtype=model_dtype)
        with torch.inference_mode():
            output = self.model(batch)
        if isinstance(output, dict):
            output = output.get("logits", output.get("pred"))
        if isinstance(output, (tuple, list)):
            output = output[-1]
        if not isinstance(output, torch.Tensor):
            raise RuntimeError("BiRefNet returned no segmentation tensor.")
        mask_tensor = output
        if mask_tensor.ndim == 4:
            mask_tensor = mask_tensor[:, :1]
        mask_tensor = functional.interpolate(
            mask_tensor.float(),
            size=(image.height, image.width),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        mask = (mask_tensor.sigmoid().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        mask_image = Image.fromarray(mask, mode="L")
        threshold = int(float(self.settings["foreground_mask_threshold"]) * 255)
        mask_fraction = float(np.count_nonzero(mask >= threshold)) / (
            image.width * image.height
        )
        if not (
            float(self.settings["minimum_foreground_fraction"])
            <= mask_fraction
            <= float(self.settings["maximum_foreground_fraction"])
        ):
            raise RuntimeError(
                f"BiRefNet mask foreground fraction {mask_fraction:.4f} is unusable."
            )
        binary_mask = mask_image.point(lambda value: 255 if value >= threshold else 0)
        box = binary_mask.getbbox()
        if box is None:
            raise RuntimeError("BiRefNet produced an empty foreground mask.")
        rgba = image.convert("RGBA")
        rgba.putalpha(mask_image)
        composite = Image.new("RGB", image.size, (127, 127, 127))
        composite.paste(image.convert("RGB"), mask=mask_image)
        color_image = composite.crop(box)
        if color_image.width < 2 or color_image.height < 2:
            raise RuntimeError("BiRefNet foreground crop is too small to classify.")
        return SegmentPrediction(rgba, color_image)


class OpenAIClipColorClassifier:
    def __init__(self, settings: dict[str, Any], device: str) -> None:
        self.settings = settings
        self.device = device
        self.model: Any = None
        self.preprocess: Any = None
        self.tokens: Any = None

    def _load(self) -> None:
        if self.model is None:
            import clip

            self.model, self.preprocess = clip.load(
                self.settings["model_id"], device=self.device
            )
            self.model.eval()
            self.tokens = clip.tokenize(list(self.settings["prompts"].values())).to(
                self.device
            )

    def predict(self, image: Image.Image) -> dict[str, float]:
        self._load()
        image_tensor = (
            self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        )
        with torch.inference_mode():
            logits, _ = self.model(image_tensor, self.tokens)
            scores = torch.softmax(logits, dim=-1)[0].cpu().tolist()
        return {
            color: float(score)
            for color, score in zip(self.settings["prompts"], scores, strict=True)
        }


def crop_species_box(
    image: Image.Image, bbox: tuple[float, float, float, float] | None
) -> Image.Image | None:
    if bbox is None:
        return None
    crop = image.crop(_crop_box(bbox, image.size))
    return crop if crop.width >= 2 and crop.height >= 2 else None
