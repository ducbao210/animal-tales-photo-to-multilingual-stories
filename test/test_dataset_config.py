import unittest

from scripts.animal_dataset import DEFAULT_CONFIG, _load_settings


class TestAnimalDatasetConfig(unittest.TestCase):
    def test_dataset_generator_uses_shared_pipeline_settings(self) -> None:
        project_root, settings = _load_settings(DEFAULT_CONFIG)

        self.assertEqual(settings["output"], project_root / "data" / "raw")
        self.assertEqual(
            settings["species"], ["cat", "owl", "hare", "dog", "squirrel"]
        )
        self.assertEqual(settings["unknown_species"], "unknown")
        self.assertEqual(settings["image_extensions"], {".jpg", ".jpeg", ".png"})
        self.assertEqual(settings["source"], "iamsouravbanerjee/animal-image-dataset-90-different-animals")
        self.assertEqual(settings["remove_images"]["unknown"], ["koala_15"])


if __name__ == "__main__":
    unittest.main()
