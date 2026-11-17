from __future__ import annotations

import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from src.display_ui.gallery import launch_streamlit_gallery, load_story_records


class TestStoryGalleryData(unittest.TestCase):
    def test_streamlit_launcher_passes_output_paths_to_app_process(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outputs = root / "outputs"
            with patch("src.display_ui.gallery.subprocess.run") as run:
                launch_streamlit_gallery(outputs, root)
            command = run.call_args.args[0]
            environment = run.call_args.kwargs["env"]
            self.assertEqual(command[1:3], ["-m", "streamlit"])
            self.assertEqual(command[3], "run")
            self.assertEqual(environment["STORY_PIPELINE_OUTPUT_ROOT"], str(outputs.resolve()))
            self.assertEqual(environment["STORY_PIPELINE_PROJECT_ROOT"], str(root.resolve()))

    def test_loads_variable_pages_and_uses_phase_images_with_keyframe_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outputs = root / "outputs"
            photo = outputs / "photo-a"
            photo.mkdir(parents=True)
            (root / "input.jpg").write_bytes(b"input")
            (photo / "character.png").write_bytes(b"keyframe")
            (photo / "phase_beginning.png").write_bytes(b"beginning")
            (photo / "scan.json").write_text(json.dumps({"input_path": "input.jpg"}), encoding="utf-8")
            (photo / "story_en.json").write_text(json.dumps({
                "image_id": "photo-a", "title": "A Garden Day", "species": "cat", "color": "brown",
                "pages": [
                    {"page": 1, "section": "beginning", "area_id": "flower_path", "text": "First page."},
                    {"page": 2, "section": "beginning", "area_id": "flower_path", "text": "Second page."},
                    {"page": 3, "section": "problem", "area_id": "pond_bridge", "text": "Third page."},
                ],
            }), encoding="utf-8")
            (photo / "translations.json").write_text(json.dumps({"translations": {
                "vi-VN": {"locale": "vi-VN", "pages": [
                    {"page": 1, "text": "Trang một."},
                    {"page": 2, "text": "Trang hai."},
                    {"page": 3, "text": "Trang ba."},
                ]},
            }}), encoding="utf-8")

            records = load_story_records(outputs, root)

            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(record["input_path"], root / "input.jpg")
            self.assertEqual(len(record["story"]["pages"]), 3)
            self.assertEqual(record["phase_images"]["beginning"], photo / "phase_beginning.png")
            self.assertEqual(record["phase_images"]["problem"], photo / "character.png")
            self.assertEqual(record["translations"]["vi-VN"]["pages"][2]["text"], "Trang ba.")

    def test_ignores_output_folders_without_a_valid_completed_story(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            outputs = Path(temporary) / "outputs"
            (outputs / "failed-photo").mkdir(parents=True)
            records = load_story_records(outputs, Path(temporary))
            self.assertEqual(records, [])


if __name__ == "__main__":
    unittest.main()
