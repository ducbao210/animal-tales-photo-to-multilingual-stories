from __future__ import annotations

import copy
import unittest
from pathlib import Path

import yaml

from src.contracts.schema import MainProfile, StoryWorld
from src.story_generation.generator import choose_support
from src.story_generation.validation import validate_story_payload


ROOT = Path(__file__).resolve().parents[1]


class TestPartCValidation(unittest.TestCase):
    def setUp(self) -> None:
        data = yaml.safe_load((ROOT / "data/story_world.yaml").read_text(encoding="utf-8"))
        self.world = StoryWorld.from_dict(data, {"cat", "owl", "hare", "dog", "squirrel"})
        self.main = self.world.main["cat"]
        self.support = self.world.main["dog"]

    def valid_payload(self) -> dict:
        texts = [
            'Brown cat Milo walked on the Flower Path and saw a ribbon near the pond.',
            '“Small paws, big plans!” Milo said, spotting the ribbon caught by a leaf.',
            'White dog Luna ran to Pond Bridge because the ribbon had floated across the water.',
            '“Together, we find a way!” Luna said, pointing to a safe stepping stone.',
            'Milo made a careful plan, and Luna shared helpful tasks.',
            'Together they used Milo’s pattern-spotting and Luna’s teamwork to reach the ribbon.',
            'They carried the ribbon back to Picnic Meadow and tied it to the blanket.',
            'Milo and Luna smiled in the sunny garden and shared a happy picnic.',
        ]
        sections = [section for section in ("beginning", "problem", "solution", "happy_ending") for _ in range(2)]
        areas = ["flower_path", "flower_path", "pond_bridge", "pond_bridge", "pond_bridge", "pond_bridge", "picnic_meadow", "picnic_meadow"]
        return {"title": "Milo Finds the Ribbon", "pages": [{"page": n, "section": sections[n-1], "area_id": areas[n-1], "text": texts[n-1]} for n in range(1, 9)]}

    def test_accepts_eight_pages_and_two_configured_areas(self) -> None:
        issues = validate_story_payload(self.valid_payload(), image_id="a", species="cat", color="brown", main=self.main, support=self.support, support_color="white", world=self.world, allowed_areas=set(self.world.areas_by_id()))
        self.assertEqual(issues, [])

    def test_rejects_long_page_missing_slogan(self) -> None:
        payload = self.valid_payload()
        payload["pages"][0]["text"] = "Brown cat Milo " + "word " * 41
        payload["pages"][1]["text"] = "Milo noticed a clue."
        issues = validate_story_payload(payload, image_id="a", species="cat", color="brown", main=self.main, support=self.support, support_color="white", world=self.world, allowed_areas=set(self.world.areas_by_id()))
        codes = {item["code"] for item in issues}
        self.assertIn("word_limit", codes)
        self.assertIn("slogan_main", codes)

    def test_support_selection_is_deterministic_and_not_main_species(self) -> None:
        first = choose_support("same-image", self.world, self.main, "brown", {"black", "white", "brown", "orange", "gray"})
        second = choose_support("same-image", self.world, self.main, "brown", {"black", "white", "brown", "orange", "gray"})
        self.assertEqual(first, second)
        self.assertNotEqual(first[0].species, self.main.species)
        self.assertNotEqual(first[1], "brown")

    def test_support_overrides_fall_back_when_same_as_main(self) -> None:
        profile, color = choose_support("same-image", self.world, self.main, "brown", {"black", "white", "brown", "orange", "gray"}, "cat", "brown")
        self.assertNotEqual(profile.species, "cat")
        self.assertNotEqual(color, "brown")

    def test_support_overrides_are_used_when_valid(self) -> None:
        profile, color = choose_support("same-image", self.world, self.main, "brown", {"black", "white", "brown", "orange", "gray"}, "dog", "white")
        self.assertEqual(profile.species, "dog")
        self.assertEqual(color, "white")

    def test_invalid_support_overrides_fall_back_to_distinct_choices(self) -> None:
        profile, color = choose_support("same-image", self.world, self.main, "brown", {"black", "white", "brown", "orange", "gray"}, "fox", "purple")
        self.assertNotEqual(profile.species, "cat")
        self.assertNotEqual(color, "brown")


if __name__ == "__main__":
    unittest.main()
