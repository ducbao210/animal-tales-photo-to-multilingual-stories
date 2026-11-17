from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.character_generation.generator import generate_character, render_prompt
from src.contracts.schema import MainProfile, StageError, validate_scan
import yaml


ROOT = Path(__file__).resolve().parents[1]


class FakeImageClient:
    def __init__(self) -> None:
        self.calls = 0
        buffer = io.BytesIO()
        Image.new("RGB", (512, 512), (220, 130, 45)).save(buffer, format="JPEG")
        self.data = buffer.getvalue()

    def generate(self, prompt: str, *, output_dir: Path, settings: dict) -> object:
        self.calls += 1
        self.prompt = prompt
        return type("Response", (), {"data": self.data, "mime_type": "image/jpeg", "model": "fake-model", "request_id": "req-1", "seed": None, "seed_supported": False})()


class TestPartB(unittest.TestCase):
    def test_prompt_renderer_requires_exact_variables(self) -> None:
        self.assertEqual(render_prompt("A {species} in {color}", {"species": "cat", "color": "brown"}), "A cat in brown")
        with self.assertRaises(ValueError):
            render_prompt("A {species}", {"species": "cat", "color": "brown"})

    def test_character_generation_reencodes_png_and_records_provider_metadata(self) -> None:
        world_data = yaml.safe_load((ROOT / "data/story_world.yaml").read_text(encoding="utf-8"))
        profile = MainProfile.from_dict({"species": "cat", **world_data["main"]["cat"]})
        scan = validate_scan({"image_id": "cat-a1", "input_path": "cat/a.jpg", "species": "cat", "color": "brown", "status": "complete", "species_uncertain": False, "color_uncertain": False}, {"cat"}, {"brown"})
        client = FakeImageClient()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = generate_character(scan, profile, {
                "allowed_species": {"cat"}, "allowed_colors": {"brown"}, "backend": "google", "model_id": "fake-model",
                "prompt_path": str(ROOT / "src/prompts/part_b/character.md"), "aspect_ratio": "1:1", "image_size": "1K",
                "max_attempts": 1,
            }, root, client)
            self.assertTrue((root / "character.png").is_file())
            with Image.open(root / "character.png") as image:
                self.assertEqual(image.format, "PNG")
                self.assertEqual(image.size, (512, 512))
            self.assertEqual(result.mime_type, "image/png")
            self.assertEqual(result.request_id, "req-1")
            self.assertFalse(result.parameters["seed_supported"])
            self.assertEqual(client.calls, 1)

    def test_unknown_species_fails_without_calling_image_provider(self) -> None:
        world_data = yaml.safe_load((ROOT / "data/story_world.yaml").read_text(encoding="utf-8"))
        profile = MainProfile.from_dict({"species": "cat", **world_data["main"]["cat"]})
        scan = validate_scan({"image_id": "unknown-a1", "input_path": "x.jpg", "species": "unknown", "color": "brown", "status": "complete", "species_uncertain": True, "color_uncertain": False}, {"cat", "unknown"}, {"brown"})
        client = FakeImageClient()
        with tempfile.TemporaryDirectory() as temporary, self.assertRaises(StageError):
            generate_character(scan, profile, {"allowed_species": {"cat"}, "allowed_colors": {"brown"}, "prompt_path": str(ROOT / "src/prompts/part_b/character.md")}, Path(temporary), client)
        self.assertEqual(client.calls, 0)


if __name__ == "__main__":
    unittest.main()
