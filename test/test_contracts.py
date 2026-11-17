from __future__ import annotations

import copy
import unittest
from pathlib import Path

import yaml

from src.contracts.schema import ContractError, StoryWorld, ensure_under, validate_scan


ROOT = Path(__file__).resolve().parents[1]


class TestSharedContracts(unittest.TestCase):
    def test_scan_preserves_unknown_species_and_optional_color(self) -> None:
        scan = validate_scan({
            "schema_version": "1.0", "image_id": "unknown-a1", "input_path": "photo.jpg",
            "species": "unknown", "color": None, "status": "partial",
            "species_uncertain": True, "color_uncertain": True, "errors": [],
        }, {"cat", "owl", "hare", "dog", "squirrel"}, {"black", "white", "brown", "orange", "gray"})
        self.assertEqual(scan.species, "unknown")
        self.assertIsNone(scan.color)
        self.assertEqual(scan.status, "partial")

    def test_scan_rejects_unconfigured_species_and_color(self) -> None:
        base = {"image_id": "x", "input_path": "x.jpg", "species": "cat", "color": "black", "status": "complete", "species_uncertain": False, "color_uncertain": False}
        with self.assertRaises(ContractError):
            validate_scan({**base, "species": "fox"}, {"cat"}, {"black"})
        with self.assertRaises(ContractError):
            validate_scan({**base, "color": "purple"}, {"cat"}, {"black"})

    def test_story_world_has_five_main_profiles_and_four_localized_areas(self) -> None:
        data = yaml.safe_load((ROOT / "data/story_world.yaml").read_text(encoding="utf-8"))
        world = StoryWorld.from_dict(data, {"cat", "owl", "hare", "dog", "squirrel"})
        self.assertEqual(len(world.main), 5)
        self.assertEqual(len(world.areas_by_id()), 4)
        self.assertEqual(set(world.areas_by_id()["flower_path"]["names"]), {"en", "fr-FR", "es-ES", "vi-VN"})

    def test_story_world_rejects_duplicate_area_ids(self) -> None:
        data = yaml.safe_load((ROOT / "data/story_world.yaml").read_text(encoding="utf-8"))
        duplicate = copy.deepcopy(data)
        duplicate["location"]["areas"][1]["id"] = duplicate["location"]["areas"][0]["id"]
        with self.assertRaises(ContractError):
            StoryWorld.from_dict(duplicate, {"cat", "owl", "hare", "dog", "squirrel"})

    def test_artifact_path_must_remain_under_photo_root(self) -> None:
        root = ROOT / "outputs" / "photo"
        self.assertEqual(ensure_under(root / "character.png", root), (root / "character.png").resolve())
        with self.assertRaises(ContractError):
            ensure_under(root.parent / "other.json", root)


if __name__ == "__main__":
    unittest.main()
