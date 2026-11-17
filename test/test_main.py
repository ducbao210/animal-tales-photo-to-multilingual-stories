from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
import yaml

import main
from src.contracts.schema import CharacterResult, StoryPage, StoryResult, StoryWorld, StageError
from src.utils.files import atomic_write_json


ROOT = Path(__file__).resolve().parents[1]


class FakeConfig:
    def __init__(self, input_dir: Path, output_dir: Path) -> None:
        self.project_root = ROOT
        self.values = {"image_extensions": [".jpg"]}
        self.taxonomy = {"species": ["cat", "owl", "hare", "dog", "squirrel"], "colors": ["black", "white", "brown", "orange", "gray"], "unknown_species": "unknown"}
        self.paths = {"input_dir": input_dir, "results_dir": output_dir / "part_a"}


class FakePartA:
    def __init__(self, _config: FakeConfig, refresh_speciesnet: bool = False) -> None:
        pass

    def _image_id(self, path: Path) -> str:
        return path.stem

    def process_one(self, path: Path) -> dict:
        return {"schema_version": "1.0", "image_id": path.stem, "input_path": path.name, "species": "cat", "color": "brown", "status": "complete", "species_uncertain": False, "color_uncertain": False, "errors": []}


class TestMainPipeline(unittest.TestCase):
    def test_partial_translation_error_is_classified_as_partial(self) -> None:
        error = StageError("D", "partial_translation", False, "es-ES is unfinished")
        self.assertEqual(main._stage_error_status(error), "partial")

    def test_context_image_failure_still_reaches_translation(self) -> None:
        world_data = yaml.safe_load((ROOT / "data/story_world.yaml").read_text(encoding="utf-8"))
        world = StoryWorld.from_dict(world_data, {"cat", "owl", "hare", "dog", "squirrel"})
        document = copy.deepcopy(yaml.safe_load((ROOT / "configs.yaml").read_text(encoding="utf-8")))
        with tempfile.TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            input_dir = tmp / "input"
            input_dir.mkdir()
            photo = input_dir / "cat.jpg"
            Image.new("RGB", (20, 20), "white").save(photo)
            gemini_client = object()
            groq_client = object()
            document["part_e"].update({
                "output_dir": str(tmp / "outputs"),
                "log_dir": str(tmp / "logs"),
                "world_path": str(ROOT / "data/story_world.yaml"),
            })

            def character(scan, profile, settings, output_dir, client, cache_dir=None):
                Image.new("RGB", (512, 512), "brown").save(output_dir / "character.png")
                result = CharacterResult(
                    "1.0", scan.image_id, "cat", "brown", "cat", profile.name,
                    "character.png", "fake", "fake",
                    {"aspect_ratio": "1:1", "image_size": "1K", "seed": None, "seed_supported": False},
                    "0" * 64, "fake-request", 512, 512, "image/png",
                )
                atomic_write_json(output_dir / "character.json", result.to_dict())
                return result

            def story(scan, character_value, world_value, settings, output_dir, client):
                sections = [section for section in ("beginning", "problem", "solution", "happy_ending") for _ in range(2)]
                pages = tuple(StoryPage(i, sections[i-1], "flower_path", f"Page {i}.") for i in range(1, 9))
                result = StoryResult(
                    "1.0", scan.image_id, "Story", "en", "cat", "cat", "brown",
                    "dog", "white", "sunny_garden", pages,
                    {"passed": True, "issues": [], "rounds": 0}, "fake", "1" * 64,
                )
                atomic_write_json(output_dir / "story_en.json", result.to_dict())
                atomic_write_json(output_dir / "page_illustration_prompts.json", [{"page": i} for i in range(1, 9)])
                return result

            def translation(story_value, world_value, settings, output_dir, client, reviewer_client=None, **_kwargs):
                pages = [{"page": i, "text": f"Page {i}."} for i in range(1, 9)]
                bundle = {
                    "schema_version": "1.0", "image_id": story_value.image_id,
                    "source_story_sha256": "2" * 64,
                    "translations": {
                        locale: {"locale": locale, "pages": pages, "validation": {"passed": True}}
                        for locale in ("fr-FR", "es-ES", "vi-VN")
                    },
                    "validation": {"passed": True},
                }
                atomic_write_json(output_dir / "translations.json", bundle)
                for locale in ("fr-FR", "es-ES", "vi-VN"):
                    atomic_write_json(output_dir / f"{locale}.json", bundle["translations"][locale])
                return bundle

            with patch.object(main, "load_pipeline_config", return_value=(ROOT, document, world)), \
                 patch.object(main, "load_config", return_value=FakeConfig(input_dir, tmp / "outputs")), \
                 patch.object(main, "discover_images", return_value=[photo]), \
                 patch.object(main, "PartAPipeline", FakePartA), \
                 patch.object(main, "create_image_client", return_value=object()), \
                 patch.object(main, "GeminiStoryClient", return_value=gemini_client), \
                 patch.object(main, "GroqTranslationClient", return_value=groq_client) as create_groq_client, \
                 patch.object(main, "generate_character", side_effect=character), \
                 patch.object(main, "generate_story", side_effect=story), \
                 patch.object(main, "translate_story", side_effect=translation) as translate, \
                 patch("src.character_generation.context_images.generate_context_images",
                       side_effect=StageError("B", "unsupported_input", False, "no reference images")):
                status = main.run_pipeline(ROOT / "configs.yaml", input_dir, generate_context_images=True)

            self.assertEqual(status, 1)
            translate.assert_called_once()
            self.assertIs(translate.call_args.kwargs["reviewer_client"], groq_client)
            self.assertEqual(create_groq_client.call_args.kwargs["requests_per_minute"], 30)
            self.assertEqual(create_groq_client.call_args.kwargs["tokens_per_minute"], 8000)
            photo_root = tmp / "outputs" / "cat"
            self.assertTrue((photo_root / "translations.json").is_file())
            failure = yaml.safe_load((photo_root / "failure.json").read_text(encoding="utf-8"))
            self.assertEqual(failure["stage"], "C_images")

    def test_display_opens_gallery_from_configured_output_after_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "configs.yaml"
            config.write_text("part_e:\n  output_dir: results\n", encoding="utf-8")
            with patch.object(main, "configure_logging"), \
                 patch.object(main, "run_pipeline", return_value=1), \
                 patch.object(main, "launch_streamlit_gallery") as open_gallery:
                status = main.main(["--config", str(config), "--display"])
            self.assertEqual(status, 1)
            open_gallery.assert_called_once_with(root / "results", root)

    def test_cli_passes_support_area_and_image_options(self) -> None:
        with patch.object(main, "configure_logging"), patch.object(main, "run_pipeline", return_value=0) as run:
            status = main.main([
                "--limit", "2",
                "--support-species", "dog", "--support-color", "orange",
                "--areas", "flower_path", "pond_bridge", "--generate-context-images",
            ])
        self.assertEqual(status, 0)
        self.assertEqual(run.call_args.args[2], 2)
        self.assertEqual(run.call_args.args[4:], (None, "dog", "orange", ["flower_path", "pond_bridge"], True, False))
    def test_one_photo_failure_does_not_block_another_photo(self) -> None:
        world_data = yaml.safe_load((ROOT / "data/story_world.yaml").read_text(encoding="utf-8"))
        world = StoryWorld.from_dict(world_data, {"cat", "owl", "hare", "dog", "squirrel"})
        document = yaml.safe_load((ROOT / "configs.yaml").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            input_dir = tmp / "input"
            input_dir.mkdir()
            photos = [input_dir / "fail.jpg", input_dir / "ok.jpg"]
            for photo in photos:
                Image.new("RGB", (20, 20), "white").save(photo)
            document = copy.deepcopy(document)
            document["part_e"]["output_dir"] = str(tmp / "outputs")
            document["part_e"]["log_dir"] = str(tmp / "logs")
            document["part_e"]["world_path"] = str(ROOT / "data/story_world.yaml")
            character_image = tmp / "character.png"
            Image.new("RGB", (512, 512), "brown").save(character_image)
            character_calls = 0
            story_calls = 0

            def fake_character(scan, profile, settings, output_dir, client, cache_dir=None):
                nonlocal character_calls
                character_calls += 1
                if scan.image_id == "fail":
                    raise StageError("B", "provider_error", False, "simulated per-photo failure")
                (output_dir / "character.png").write_bytes(character_image.read_bytes())
                result = CharacterResult("1.0", scan.image_id, "cat", "brown", "cat", profile.name, "character.png", "fake", "fake", {"aspect_ratio": "1:1", "image_size": "1K", "seed": None, "seed_supported": False}, "0" * 64, "fake-request", 512, 512, "image/png")
                atomic_write_json(output_dir / "character.json", result.to_dict())
                return result

            def fake_story(scan, character, world_value, settings, output_dir, client):
                nonlocal story_calls
                story_calls += 1
                sections = [section for section in ("beginning", "problem", "solution", "happy_ending") for _ in range(2)]
                pages = tuple(StoryPage(i, sections[i-1], "flower_path" if i < 5 else "pond_bridge", f"Page {i}.") for i in range(1, 9))
                result = StoryResult("1.0", scan.image_id, "Story", "en", "cat", "cat", "brown", "dog", "white", "sunny_garden", pages, {"passed": True, "issues": [], "rounds": 0}, "fake", "1" * 64)
                atomic_write_json(output_dir / "story_en.json", result.to_dict())
                atomic_write_json(output_dir / "page_illustration_prompts.json", [{"page": i} for i in range(1, 9)])
                return result

            def fake_translation(story, world_value, settings, output_dir, client, reviewer_client=None, **_kwargs):
                pages = [{"page": i, "text": f"Page {i}."} for i in range(1, 9)]
                bundle = {"schema_version": "1.0", "image_id": story.image_id, "source_story_sha256": "2" * 64,
                          "translations": {locale: {"locale": locale, "pages": pages, "validation": {"passed": True}} for locale in ("fr-FR", "es-ES", "vi-VN")},
                          "validation": {"passed": True}}
                atomic_write_json(output_dir / "translations.json", bundle)
                for locale in ("fr-FR", "es-ES", "vi-VN"):
                    atomic_write_json(output_dir / f"{locale}.json", bundle["translations"][locale])
                return bundle

            with patch.object(main, "load_pipeline_config", return_value=(ROOT, document, world)), \
                 patch.object(main, "load_config", return_value=FakeConfig(input_dir, tmp / "outputs")), \
                 patch.object(main, "discover_images", return_value=photos), \
                 patch.object(main, "PartAPipeline", FakePartA), \
                 patch.object(main, "create_image_client", return_value=object()), \
                 patch.object(main, "GeminiStoryClient", return_value=object()), \
                 patch.object(main, "GroqTranslationClient", return_value=object()), \
                 patch.object(main, "generate_character", side_effect=fake_character), \
                 patch.object(main, "generate_story", side_effect=fake_story), \
                 patch.object(main, "translate_story", side_effect=fake_translation):
                status = main.run_pipeline(ROOT / "configs.yaml", input_dir, limit=2)
                resumed_status = main.run_pipeline(ROOT / "configs.yaml", input_dir, limit=2)

            self.assertEqual(status, 1)
            self.assertEqual(resumed_status, 1)
            self.assertEqual(character_calls, 3)  # failed photo is retried; completed photo is reused
            self.assertEqual(story_calls, 1)
            self.assertTrue((tmp / "outputs" / "fail" / "failure.json").is_file())
            self.assertTrue((tmp / "outputs" / "ok" / "story_en.json").is_file())
            self.assertTrue((tmp / "outputs" / "ok" / "translations.json").is_file())


if __name__ == "__main__":
    unittest.main()
