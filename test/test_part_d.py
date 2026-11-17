from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.contracts.schema import StageError
from src.translation.translator import translate_story
from src.translation.validation import validate_translation_payload


class TestPartDValidation(unittest.TestCase):
    def test_rate_limited_locale_does_not_discard_completed_translation(self) -> None:
        pages = [SimpleNamespace(page=i, text="A quiet day.") for i in range(1, 9)]
        story = SimpleNamespace(
            image_id="photo", species="cat", support_id="dog",
            color="white", support_color="black", pages=pages,
            to_dict=lambda: {
                "image_id": "photo", "species": "cat",
                "pages": [{"page": page.page, "text": page.text} for page in pages],
            },
        )
        locales = ("fr-FR", "es-ES", "vi-VN")
        def profile(species: str) -> SimpleNamespace:
            return SimpleNamespace(
                species=species, gender="neutral",
                names={"en": species, **{locale: species for locale in locales}},
                slogans={locale: "Hello" for locale in locales},
            )
        world = SimpleNamespace(
            main={"cat": profile("cat"), "dog": profile("dog")},
            location={"areas": []}, areas_by_id=lambda: {},
            pronouns={locale: {} for locale in locales},
            punctuation={locale: "" for locale in locales},
        )
        prompt_root = Path(__file__).resolve().parents[1] / "src" / "prompts" / "part_d"
        settings = {
            "locales": list(locales), "correction_rounds": 0,
            "prompt_translate": str(prompt_root / "translation.md"),
            "prompt_review": str(prompt_root / "translation_review.md"),
            "prompt_refine": str(prompt_root / "translation_refine.md"),
        }
        def quota_limited_call(_client: object, _prompt: str, schema_name: str, *_args: object) -> dict:
            if schema_name == "translation_review_es_ES":
                raise StageError("D", "rate_limit", True, "daily quota exhausted")
            if schema_name.startswith("translation_review_"):
                return {"passed": True, "issues": []}
            return {"pages": [{"page": i, "text": "A quiet day."} for i in range(1, 9)]}

        with tempfile.TemporaryDirectory() as temporary, patch(
            "src.translation.translator._call", side_effect=quota_limited_call
        ):
            output_dir = Path(temporary)
            with self.assertRaises(StageError) as raised:
                translate_story(story, world, settings, output_dir, object())
            self.assertEqual(raised.exception.category, "partial_translation")
            bundle = json.loads((output_dir / "translations.json").read_text(encoding="utf-8"))
            self.assertEqual(set(bundle["translations"]), {"fr-FR", "vi-VN"})
            self.assertEqual(bundle["validation"]["failed_locales"], {"es-ES": "rate_limit"})
            self.assertTrue((output_dir / "fr-FR.json").is_file())
            self.assertTrue((output_dir / "vi-VN.json").is_file())

    def test_failed_french_review_does_not_block_other_valid_locales(self) -> None:
        pages = [SimpleNamespace(page=i, text="A quiet day.") for i in range(1, 9)]
        story = SimpleNamespace(
            image_id="photo", species="cat", support_id="dog",
            color="white", support_color="black", pages=pages,
            to_dict=lambda: {
                "image_id": "photo", "species": "cat",
                "pages": [{"page": page.page, "text": page.text} for page in pages],
            },
        )
        locales = ("fr-FR", "es-ES", "vi-VN")
        def profile(species: str) -> SimpleNamespace:
            return SimpleNamespace(
                species=species, gender="neutral",
                names={"en": species, **{locale: species for locale in locales}},
                slogans={locale: "Hello" for locale in locales},
            )
        world = SimpleNamespace(
            main={"cat": profile("cat"), "dog": profile("dog")},
            location={"areas": []}, areas_by_id=lambda: {},
            pronouns={locale: {} for locale in locales},
            punctuation={locale: "" for locale in locales},
        )
        prompt_root = Path(__file__).resolve().parents[1] / "src" / "prompts" / "part_d"
        settings = {
            "locales": list(locales), "correction_rounds": 0,
            "prompt_translate": str(prompt_root / "translation.md"),
            "prompt_review": str(prompt_root / "translation_review.md"),
            "prompt_refine": str(prompt_root / "translation_refine.md"),
        }
        def fake_call(_client: object, _prompt: str, schema_name: str, *_args: object) -> dict:
            if schema_name == "translation_review_fr_FR":
                return {"passed": False, "issues": [{"page": 1, "code": "meaning", "explanation": "Mismatch"}]}
            if schema_name.startswith("translation_review_"):
                return {"passed": True, "issues": []}
            return {"pages": [{"page": i, "text": "A quiet day."} for i in range(1, 9)]}

        with tempfile.TemporaryDirectory() as temporary, patch(
            "src.translation.translator._call", side_effect=fake_call
        ):
            output_dir = Path(temporary)
            with self.assertRaises(StageError) as raised:
                translate_story(story, world, settings, output_dir, object())
            self.assertEqual(raised.exception.category, "partial_translation")
            bundle = json.loads((output_dir / "translations.json").read_text(encoding="utf-8"))
            self.assertEqual(set(bundle["translations"]), {"es-ES", "vi-VN"})
            self.assertEqual(bundle["validation"]["failed_locales"], {"fr-FR": "validation_exhausted"})
            self.assertTrue((output_dir / "es-ES.json").is_file())
            self.assertTrue((output_dir / "vi-VN.json").is_file())

            def valid_call(_client: object, _prompt: str, schema_name: str, *_args: object) -> dict:
                if schema_name.startswith("translation_review_"):
                    return {"passed": True, "issues": []}
                return {"pages": [{"page": i, "text": "A quiet day."} for i in range(1, 9)]}

            with patch("src.translation.translator._call", side_effect=valid_call):
                completed = translate_story(story, world, settings, output_dir, object())
            self.assertEqual(set(completed.translations), set(locales))
            self.assertTrue(completed.validation["passed"])

    def test_translation_requires_aligned_pages_and_names(self) -> None:
        source = [{"page": i, "text": "Milo sees Bea."} for i in range(1, 9)]
        translated = {"pages": [{"page": i, "text": "Milo voit Béa."} for i in range(1, 9)]}
        issues = validate_translation_payload(source, translated, locale="fr-FR", required_names={"Milo": "Milo", "Bea": "Béa"})
        self.assertEqual(issues, [])
        translated["pages"][3]["page"] = 99
        translated["pages"][4]["text"] = "Milo voit seulement la fleur."
        issues = validate_translation_payload(source, translated, locale="fr-FR", required_names={"Milo": "Milo", "Bea": "Béa"})
        codes = {issue["code"] for issue in issues}
        self.assertIn("page_id", codes)
        self.assertIn("missing_name", codes)

    def test_spanish_and_french_punctuation_rules_are_checked(self) -> None:
        source = [{"page": i, "text": "Milo says hello."} for i in range(1, 9)]
        spanish = {"pages": [{"page": i, "text": "¿Milo viene? ¡Sí!"} for i in range(1, 9)]}
        self.assertEqual(validate_translation_payload(source, spanish, locale="es-ES", required_names={"Milo": "Milo"}), [])
        french = {"pages": [{"page": i, "text": "Milo vient!"} for i in range(1, 9)]}
        issues = validate_translation_payload(source, french, locale="fr-FR", required_names={"Milo": "Milo"})
        self.assertIn("french_spacing", {issue["code"] for issue in issues})


if __name__ == "__main__":
    unittest.main()
