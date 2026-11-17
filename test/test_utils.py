from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path

from src.utils.files import atomic_write_json
from src.utils.logger import JsonlEventLogger
from src.utils.retry import retry_call


class TestPipelineUtilities(unittest.TestCase):
    def test_run_summary_counts_partial_photos_separately(self) -> None:
        with self.assertLogs("src.utils.logger", level="INFO") as captured:
            JsonlEventLogger._write_console_event({
                "event": "run_end",
                "run_id": "run-1",
                "results": [
                    {"status": "complete"},
                    {"status": "partial"},
                    {"status": "failed"},
                ],
            })
        self.assertIn("1 complete, 1 partial, 1 failed", captured.output[0])

    def test_jsonl_event_logger_is_thread_safe_and_adds_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "logs" / "events.jsonl"
            logger = JsonlEventLogger(path)

            def write_events(worker_id: int) -> None:
                for event_id in range(20):
                    logger.write(
                        {
                            "event": "step_end",
                            "image_id": f"photo-{worker_id}",
                            "step": f"step-{event_id}",
                        }
                    )

            workers = [
                threading.Thread(target=write_events, args=(worker_id,))
                for worker_id in range(5)
            ]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join()

            records = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(records), 100)
            self.assertTrue(all(record.get("timestamp") for record in records))
            self.assertEqual(
                {record["image_id"] for record in records},
                {f"photo-{worker_id}" for worker_id in range(5)},
            )

    def test_atomic_json_write_creates_complete_utf8_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "results" / "photo.json"
            expected = {"species": "unknown", "note": "ảnh thú cưng"}
            atomic_write_json(path, expected)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), expected)
            self.assertEqual(list(path.parent.iterdir()), [path])

            replacement = {"species": "cat", "color": "gray"}
            atomic_write_json(path, replacement)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), replacement)

    def test_retry_call_retries_only_to_configured_limit(self) -> None:
        attempts = 0
        retry_events: list[int] = []

        def flaky_operation() -> str:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise OSError("temporary service failure")
            return "done"

        result = retry_call(
            flaky_operation,
            max_attempts=3,
            should_retry=lambda error: isinstance(error, OSError),
            on_retry=lambda attempt, _error: retry_events.append(attempt),
        )
        self.assertEqual(result, "done")
        self.assertEqual(attempts, 3)
        self.assertEqual(retry_events, [1, 2])

        with self.assertRaises(ValueError):
            retry_call(lambda: None, max_attempts=0)


if __name__ == "__main__":
    unittest.main()
