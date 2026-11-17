from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from src.utils.configuration import load_pipeline_config


ROOT = Path(__file__).resolve().parents[1]


class TestPromptConfiguration(unittest.TestCase):
    def test_config_uses_google_and_no_manual_banned_word_file(self) -> None:
        test_keys = {
            "GEMINI_API_KEY": "test-gemini-key",
            "REPLICATE_API_TOKEN": "test-replicate-token",
            "GROQ_API_KEY": "test-groq-key",
        }
        with patch.dict(os.environ, test_keys):
            _root, config, _world = load_pipeline_config(ROOT / "configs.yaml")
        self.assertEqual(config["part_b"]["backend"], "replicate")
        self.assertEqual(config["part_b"]["model_id"], "google/imagen-4")
        self.assertEqual(config["part_c"]["model_id"], "gemini-3.6-flash")
        self.assertEqual(config["part_d"]["model_id"], "openai/gpt-oss-120b")
        self.assertNotIn("banned_words_path", config["part_e"])
        self.assertFalse((ROOT / "data" / "banned_words.yaml").exists())

    def test_each_gen_ai_stage_points_to_a_prompt_under_src_prompts(self) -> None:
        test_keys = {
            "GEMINI_API_KEY": "test-gemini-key",
            "REPLICATE_API_TOKEN": "test-replicate-token",
            "GROQ_API_KEY": "test-groq-key",
        }
        with patch.dict(os.environ, test_keys):
            _root, config, _world = load_pipeline_config(ROOT / "configs.yaml")

        prompt_paths = [
            config["part_b"][key]
            for key in ("prompt_path",)
        ]
        prompt_paths.extend(
            config["part_c"][key]
            for key in (
                "prompt_outline",
                "prompt_write",
                "prompt_review",
                "prompt_refine",
                "prompt_phase_plan",
                "prompt_page_illustration",
                "prompt_context",
            )
        )
        prompt_paths.extend(
            config["part_d"][key]
            for key in (
                "prompt_translate",
                "prompt_review",
                "prompt_refine",
                "prompt_back_translation",
            )
        )

        self.assertEqual(len(prompt_paths), 12)
        for prompt_path in prompt_paths:
            resolved = Path(prompt_path)
            self.assertTrue(resolved.is_relative_to(ROOT / "src" / "prompts"))
            self.assertTrue(resolved.is_file(), prompt_path)


if __name__ == "__main__":
    unittest.main()
