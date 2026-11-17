from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from src.character_generation.context_images import generate_context_images, validate_phase_plans
from src.contracts.schema import StageError, StoryPage, StoryResult


class TestContextImagePlans(unittest.TestCase):
    def setUp(self) -> None:
        sections = [section for section in ("beginning", "problem", "solution", "happy_ending") for _ in range(2)]
        self.story = StoryResult(
            "1.0", "photo", "Story", "en", "cat", "cat", "brown", "dog", "white", "sunny_garden",
            tuple(StoryPage(index, sections[index - 1], "flower_path" if index < 5 else "pond_bridge", f"Page {index}.") for index in range(1, 9)),
            {"passed": True}, "gpt", "0" * 64,
        )
        self.plans = [
            {"section": section, "page_ids": [2 * index + 1, 2 * index + 2],
             "area_id": "flower_path" if index < 2 else "pond_bridge", "scene": "A clear scene.",
             "action": "The characters help.", "emotion": "hopeful", "character_ids": ["cat", "dog"] if index > 0 else ["cat"]}
            for index, section in enumerate(("beginning", "problem", "solution", "happy_ending"))
        ]

    def test_accepts_four_ordered_story_phase_plans(self) -> None:
        result = validate_phase_plans(self.plans, self.story, {"flower_path", "pond_bridge"}, {"cat", "dog"})
        self.assertEqual([plan["section"] for plan in result], ["beginning", "problem", "solution", "happy_ending"])

    def test_rejects_unselected_area_and_unknown_character(self) -> None:
        plans = [dict(item) for item in self.plans]
        plans[0] = {**plans[0], "area_id": "tall_oak"}
        with self.assertRaises(StageError):
            validate_phase_plans(plans, self.story, {"flower_path", "pond_bridge"}, {"cat", "dog"})

    def test_generates_each_phase_from_metadata_without_reference_images(self) -> None:
        root = Path(__file__).resolve().parents[1]
        profiles = {
            "cat": SimpleNamespace(species="cat", name="Milo", visual_detail="white paws and a blue scarf"),
            "dog": SimpleNamespace(species="dog", name="Bea", visual_detail="black ears and a yellow collar"),
        }
        areas = {
            "flower_path": {"names": {"en": "Flower Path"}, "description": "A path by flowers."},
            "pond_bridge": {"names": {"en": "Pond Bridge"}, "description": "A bridge over a pond."},
        }
        world = SimpleNamespace(
            main=profiles,
            areas_by_id=lambda: areas,
        )

        class CapturingImageClient:
            def __init__(self) -> None:
                self.calls: list[tuple[str, dict]] = []

            def generate(self, prompt: str, *, output_dir: Path, settings: dict) -> SimpleNamespace:
                self.calls.append((prompt, settings))
                content = io.BytesIO()
                Image.new("RGB", (4, 4), "white").save(content, format="PNG")
                return SimpleNamespace(data=content.getvalue())

        image_client = CapturingImageClient()
        story_settings = {
            "selected_areas": ["flower_path", "pond_bridge"],
            "prompt_phase_plan": str(root / "src/prompts/features/context_images/phase_plan.md"),
            "prompt_context": str(root / "src/prompts/features/context_images/context_image.md"),
            "review_max_output_tokens": 700,
        }
        with tempfile.TemporaryDirectory() as temporary, patch(
            "src.character_generation.context_images._call",
            return_value={"plans": self.plans},
        ):
            outputs = generate_context_images(
                self.story,
                world,
                story_settings,
                {"model_id": "google/imagen-4"},
                Path(temporary),
                object(),
                image_client,
            )

        self.assertEqual(len(outputs), 4)
        self.assertEqual(len(image_client.calls), 4)
        for prompt, settings in image_client.calls:
            self.assertNotIn("reference_image_paths", settings)
            self.assertNotIn("reference_image_path", settings)
            self.assertNotIn("reference image", prompt.lower())
            self.assertIn("white paws and a blue scarf", prompt)
        plans = [dict(item) for item in self.plans]
        plans[1] = {**plans[1], "character_ids": ["cat", "fox"]}
        with self.assertRaises(StageError):
            validate_phase_plans(plans, self.story, {"flower_path", "pond_bridge"}, {"cat", "dog"})


if __name__ == "__main__":
    unittest.main()
