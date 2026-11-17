from __future__ import annotations

import copy
import json
import tempfile
import threading
import unittest
from unittest.mock import patch
from pathlib import Path
from typing import Any

from PIL import Image
import numpy as np
import torch
import yaml

from src.animal_scan.config import PartAConfig, load_config
from src.animal_scan.data import create_test_loader, discover_images, load_labeled_examples
from src.animal_scan.models import (
    BiRefNetSegmenter,
    SegmentPrediction,
    SpeciesPrediction,
    SpeciesNetRunner,
    map_species_label,
    parse_speciesnet_output,
)
from src.animal_scan.pipeline import PartAPipeline
from src.utils.logger import JsonlEventLogger


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs.yaml"


class FakeSpeciesNet:
    def __init__(self, fail_name: str | None = None) -> None:
        self.fail_name = fail_name

    def predict(self, image_path: Path) -> SpeciesPrediction:
        if image_path.name == self.fail_name:
            raise RuntimeError("simulated per-image SpeciesNet failure")
        return SpeciesPrediction("Felis catus", 0.91, (0.1, 0.1, 0.8, 0.8))


class ConcurrentSpeciesNet:
    def __init__(self) -> None:
        self.barrier = threading.Barrier(2)
        self.max_active = 0
        self.active = 0
        self.lock = threading.Lock()

    def predict(self, image_path: Path) -> SpeciesPrediction:
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            self.barrier.wait(timeout=5)
            return SpeciesPrediction("Felis catus", 0.91, None)
        finally:
            with self.lock:
                self.active -= 1


class FakeSegmenter:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.foreground_objects: list[Image.Image] = []

    def segment(self, image: Image.Image) -> SegmentPrediction:
        if self.fail:
            raise RuntimeError("simulated BiRefNet foreground failure")
        foreground = image.convert("RGBA")
        color_image = image.crop((0, 0, max(2, image.width // 2), image.height))
        self.foreground_objects.append(color_image)
        return SegmentPrediction(foreground, color_image)


class FakeClip:
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.images: list[Image.Image] = []

    def predict(self, image: Image.Image) -> dict[str, float]:
        self.images.append(image)
        return self.scores


class BrokenClip:
    def predict(self, image: Image.Image) -> dict[str, float]:
        raise RuntimeError("simulated CLIP weight failure")


def test_config(root: Path, output_root: Path) -> PartAConfig:
    config = load_config(CONFIG_PATH)
    values = copy.deepcopy(config.values)
    values["paths"]["results_dir"] = str(output_root / "results")
    values["paths"]["foreground_dir"] = str(output_root / "foreground")
    values["paths"]["log_file"] = str(output_root / "logs" / "part_a.jsonl")
    values["retries"]["max_attempts"] = 2
    values["retries"]["retry_delay_seconds"] = 0
    return PartAConfig(root, values)


def confident_scores() -> dict[str, float]:
    return {"black": 0.05, "white": 0.03, "orange": 0.07, "gray": 0.05, "brown": 0.8}


class TestPartA(unittest.TestCase):
    def test_speciesnet_runner_can_force_redownload_of_cached_model(self) -> None:
        runner = SpeciesNetRunner(
            {
                "model_id": "kaggle:google/speciesnet/pyTorch/v4.0.3a/1",
                "command": ["python", "-m", "speciesnet.scripts.run_model"],
                "timeout_seconds": 5,
            },
            force_model_download=True,
        )
        with tempfile.TemporaryDirectory() as temporary:
            image_path = Path(temporary) / "animal.jpg"
            image_path.write_bytes(b"image")
            forced_flags: list[bool] = []

            def run_command(command: list[str], **_kwargs: Any) -> Any:
                forced = "--force_model_download" in command
                forced_flags.append(forced)
                output_path = Path(command[command.index("--predictions_json") + 1])
                output_path.write_text(
                    json.dumps(
                        {
                            "predictions": [
                                {
                                    "classifications": {"classes": ["Felis catus"]},
                                    "detections": [
                                        {"category": "1", "conf": 0.9, "bbox": [0, 0, 1, 1]}
                                    ],
                                }
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
                return type("Completed", (), {"returncode": 0, "stderr": "", "stdout": ""})()

            with patch("src.animal_scan.models.subprocess.run", side_effect=run_command):
                prediction = runner.predict(image_path)
                runner.predict(image_path)

        self.assertEqual(prediction.label, "Felis catus")
        self.assertEqual(forced_flags, [True, False])

    def test_birefnet_input_uses_model_parameter_dtype(self) -> None:
        config = load_config(CONFIG_PATH)
        settings = copy.deepcopy(config.values["birefnet"])
        settings["image_size"] = 8
        segmenter = BiRefNetSegmenter(settings, "cpu")

        class HalfPrecisionModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones((), dtype=torch.float16))
                self.input_dtype: torch.dtype | None = None

            def forward(self, batch: torch.Tensor) -> tuple[torch.Tensor]:
                self.input_dtype = batch.dtype
                logits = torch.full((1, 1, 8, 8), -10.0, dtype=batch.dtype)
                logits[:, :, 2:6, 2:6] = 10.0
                return (logits,)

        model = HalfPrecisionModel()
        segmenter.model = model
        segmenter.segment(Image.new("RGB", (8, 8), (120, 80, 40)))

        self.assertEqual(model.input_dtype, torch.float16)

    def test_configured_taxonomy_drives_labeled_data_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            document = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
            part_a = document["part_a"]
            part_a["taxonomy"]["species"] = ["fox"]
            part_a["taxonomy"]["species_aliases"] = {"fox": ["fox"]}
            part_a["taxonomy"]["colors"] = ["tan"]
            part_a["clip"]["prompts"] = {"tan": "a tan fox"}
            part_a["paths"]["labels_csv"] = "labels.csv"
            Image.new("RGB", (2, 2)).save(root / "fox.png")
            (root / "labels.csv").write_text(
                "relative_path,species,color\nfox.png,fox,tan\n", encoding="utf-8"
            )
            config_path = root / "configs.yaml"
            config_path.write_text(yaml.safe_dump(document), encoding="utf-8")

            examples = load_labeled_examples(load_config(config_path))

            self.assertEqual(len(examples), 1)
            self.assertEqual((examples[0].species, examples[0].color), ("fox", "tan"))

    def test_config_taxonomy_and_existing_labels(self) -> None:
        config = load_config(CONFIG_PATH)
        self.assertEqual(
            config.taxonomy["species"], ["cat", "owl", "hare", "dog", "squirrel"]
        )
        self.assertEqual(
            config.taxonomy["colors"], ["black", "white", "brown", "orange", "gray"]
        )
        self.assertEqual(config.taxonomy["unknown_species"], "unknown")
        self.assertEqual(len(load_labeled_examples(config)), 356)
        batches = list(create_test_loader(config, batch_size=100))
        self.assertEqual(sum(len(batch["image_path"]) for batch in batches), 356)
        self.assertEqual(batches[0]["species"][0], "cat")

    def test_species_mapping_requires_an_exact_supported_alias(self) -> None:
        config = load_config(CONFIG_PATH)
        aliases = config.taxonomy["species_aliases"]
        self.assertEqual(map_species_label("Felis_catus", aliases), ("cat", None))
        self.assertEqual(map_species_label("Felidae", aliases)[0], "unknown")
        self.assertEqual(map_species_label("rabbit", aliases)[0], "unknown")
        self.assertEqual(map_species_label("blank", aliases)[0], "unknown")
        self.assertEqual(map_species_label(None, aliases)[0], "unknown")

    def test_speciesnet_output_requires_an_animal_detection(self) -> None:
        output = {
            "predictions": [
                {
                    "detections": [
                        {"category": "1", "conf": 0.9, "bbox": [0.1, 0.2, 0.7, 0.6]}
                    ],
                    "classifications": {
                        "classes": ["Felis catus", "Felidae"],
                        "scores": [0.9, 0.1],
                    },
                }
            ]
        }
        prediction = parse_speciesnet_output(output)
        self.assertEqual(prediction.label, "Felis catus")
        self.assertEqual(prediction.confidence, 0.9)
        self.assertEqual(prediction.bounding_box, (0.1, 0.2, 0.7, 0.6))
        output["predictions"][0]["detections"][0]["category"] = "3"
        self.assertIsNone(parse_speciesnet_output(output).label)

    def test_birefnet_foreground_crop_excludes_background_pixels(self) -> None:
        config = load_config(CONFIG_PATH)
        settings = copy.deepcopy(config.values["birefnet"])
        settings["image_size"] = 8
        segmenter = BiRefNetSegmenter(settings, "cpu")

        class FixedMask:
            def __call__(self, batch: torch.Tensor) -> tuple[torch.Tensor]:
                logits = torch.full((1, 1, 8, 8), -10.0)
                logits[:, :, 2:6, 2:6] = 10.0
                return (logits,)

        segmenter.model = FixedMask()
        pixels = np.zeros((80, 80, 3), dtype=np.uint8)
        pixels[:, :] = (0, 0, 255)
        pixels[20:60, 20:60] = (255, 0, 0)
        image = Image.fromarray(pixels, mode="RGB")
        output = segmenter.segment(image)

        self.assertLessEqual(output.color_image.width, 41)
        center_pixel = output.color_image.getpixel(
            (output.color_image.width // 2, output.color_image.height // 2)
        )
        self.assertGreater(center_pixel[0], 250)
        self.assertEqual(center_pixel[1:], (0, 0))
        self.assertFalse(
            any(blue > red and blue > green for red, green, blue in output.color_image.getdata())
        )

    def test_three_photos_process_independently_in_required_order(self) -> None:
        paths = list(discover_images(ROOT / "data" / "raw", (".jpg",)))[:3]
        self.assertEqual(len(paths), 3)
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            config = test_config(ROOT, output_root)
            segmenter = FakeSegmenter()
            clip = FakeClip(confident_scores())
            pipeline = PartAPipeline(
                config,
                speciesnet=FakeSpeciesNet(fail_name=paths[1].name),
                birefnet=segmenter,
                clip_classifier=clip,
                event_logger=JsonlEventLogger(config.paths["log_file"]),
            )
            originals = {path: path.read_bytes() for path in paths}
            results = pipeline.process_paths(paths)

            self.assertEqual(
                [result["species"] for result in results], ["cat", "unknown", "cat"]
            )
            self.assertEqual([result["color"] for result in results], ["brown"] * 3)
            self.assertEqual(
                [result["status"] for result in results],
                ["complete", "partial", "complete"],
            )
            self.assertEqual(len(clip.images), 3)
            self.assertEqual(
                {id(image) for image in clip.images},
                {id(image) for image in segmenter.foreground_objects},
            )
            self.assertTrue(all(path.read_bytes() == originals[path] for path in paths))
            result_files = list(config.paths["results_dir"].glob("*.json"))
            self.assertEqual(len(result_files), 3)
            for result_file in result_files:
                self.assertEqual(json.loads(result_file.read_text())["schema_version"], "1.0")

            events = [
                json.loads(line)
                for line in config.paths["log_file"].read_text(encoding="utf-8").splitlines()
            ]
            first_image_events = [
                event for event in events if event["image_id"] == results[0]["image_id"]
            ]
            completed_steps = [
                event["step"]
                for event in first_image_events
                if event["event"] == "step_end" and event["status"] == "succeeded"
            ]
            self.assertLess(
                completed_steps.index("speciesnet"), completed_steps.index("birefnet")
            )
            self.assertLess(
                completed_steps.index("birefnet"), completed_steps.index("clip_color")
            )
            retry_events = [event for event in events if event["event"] == "step_retry"]
            self.assertEqual(len(retry_events), 1)
            self.assertEqual(retry_events[0]["step"], "speciesnet")

    def test_photo_processing_uses_bounded_concurrency(self) -> None:
        paths = list(discover_images(ROOT / "data" / "raw", (".jpg",)))[:3]
        with tempfile.TemporaryDirectory() as temporary:
            config = test_config(ROOT, Path(temporary))
            speciesnet = ConcurrentSpeciesNet()
            pipeline = PartAPipeline(
                config,
                speciesnet=speciesnet,
                birefnet=FakeSegmenter(),
                clip_classifier=FakeClip(confident_scores()),
                event_logger=JsonlEventLogger(config.paths["log_file"]),
            )
            results = pipeline.process_paths(paths)
            self.assertEqual(len(results), 3)
            self.assertGreaterEqual(speciesnet.max_active, 2)
            self.assertEqual(
                [result["input_path"] for result in results],
                [pipeline._display_path(path) for path in paths],
            )

    def test_foreground_failure_uses_fallback_and_marks_color_uncertain(self) -> None:
        path = next(iter(discover_images(ROOT / "data" / "raw", (".jpg",))))
        ambiguous = {
            "black": 0.24,
            "white": 0.23,
            "orange": 0.22,
            "gray": 0.16,
            "brown": 0.15,
        }
        with tempfile.TemporaryDirectory() as temporary:
            config = test_config(ROOT, Path(temporary))
            clip = FakeClip(ambiguous)
            pipeline = PartAPipeline(
                config,
                speciesnet=FakeSpeciesNet(),
                birefnet=FakeSegmenter(fail=True),
                clip_classifier=clip,
                event_logger=JsonlEventLogger(config.paths["log_file"]),
            )
            result = pipeline.process_one(path)
            self.assertIn(result["color"], config.taxonomy["colors"])
            self.assertEqual(result["color"], "black")
            self.assertTrue(result["color_uncertain"])
            self.assertEqual(result["foreground_status"], "fallback")
            self.assertIn("fallback", result["color_uncertainty_reason"])
            self.assertTrue(result["foreground_path"])
            self.assertEqual(result["status"], "partial")
            self.assertIn("birefnet", {error["step"] for error in result["errors"]})

    def test_clip_failure_does_not_fabricate_a_color(self) -> None:
        path = next(iter(discover_images(ROOT / "data" / "raw", (".jpg",))))
        with tempfile.TemporaryDirectory() as temporary:
            config = test_config(ROOT, Path(temporary))
            pipeline = PartAPipeline(
                config,
                speciesnet=FakeSpeciesNet(),
                birefnet=FakeSegmenter(),
                clip_classifier=BrokenClip(),
                event_logger=JsonlEventLogger(config.paths["log_file"]),
            )
            result = pipeline.process_one(path)
            self.assertIsNone(result["color"])
            self.assertTrue(result["color_uncertain"])
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["errors"][-1]["step"], "clip_color")

    def test_bad_photo_does_not_block_next_photo(self) -> None:
        valid = next(iter(discover_images(ROOT / "data" / "raw", (".jpg",))))
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            broken = temporary_path / "broken.jpg"
            broken.write_text("not an image", encoding="utf-8")
            config = test_config(ROOT, temporary_path / "out")
            pipeline = PartAPipeline(
                config,
                speciesnet=FakeSpeciesNet(),
                birefnet=FakeSegmenter(),
                clip_classifier=FakeClip(confident_scores()),
                event_logger=JsonlEventLogger(config.paths["log_file"]),
            )
            results = pipeline.process_paths([broken, valid])
            self.assertEqual([result["status"] for result in results], ["failed", "complete"])
            self.assertIsNone(results[0]["color"])
            self.assertEqual(results[1]["color"], "brown")


if __name__ == "__main__":
    unittest.main()
