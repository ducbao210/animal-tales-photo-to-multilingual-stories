from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol


LOGGER = logging.getLogger(__name__)
_PART_A_STEP_LABELS = {
    "input": "load input",
    "speciesnet": "SpeciesNet classification",
    "birefnet": "BiRefNet segmentation",
    "foreground_save": "save foreground",
    "clip_color": "CLIP color classification",
}


class EventLogger(Protocol):
    def write(self, event: dict[str, Any]) -> None: ...


class JsonlEventLogger:
    """Append timestamped, thread-safe structured pipeline events."""

    def __init__(self, path: str | Path, *, console: bool = True) -> None:
        self.path = Path(path)
        self.console = console
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, event: dict[str, Any]) -> None:
        record = dict(event)
        record.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        serialized = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(serialized + "\n")
        if self.console:
            self._write_console_event(record)

    @staticmethod
    def _write_console_event(event: dict[str, Any]) -> None:
        event_name = event.get("event")
        image_id = event.get("image_id")
        step = event.get("step")
        if event_name == "run_start":
            LOGGER.info(
                "Pipeline run %s started: %s photo(s)",
                event.get("run_id", "unknown"),
                event.get("photo_count", "?"),
            )
            return
        if event_name == "run_end":
            results = event.get("results", [])
            complete = sum(
                isinstance(result, dict) and result.get("status") == "complete"
                for result in results
            ) if isinstance(results, list) else 0
            partial = sum(
                isinstance(result, dict) and result.get("status") == "partial"
                for result in results
            ) if isinstance(results, list) else 0
            failed = len(results) - complete - partial if isinstance(results, list) else 0
            LOGGER.info(
                "Pipeline run %s finished: %d complete, %d partial, %d failed",
                event.get("run_id", "unknown"), complete, partial, failed,
            )
            return
        if not image_id or not step:
            return

        step_name = str(step)
        if step_name in _PART_A_STEP_LABELS:
            part = "A"
            detail = _PART_A_STEP_LABELS[step_name]
        else:
            part = step_name[0] if step_name[0] in "ABCDE" else step_name
            detail = None
        stage_label = {"A": "Stage 1", "B": "Stage 2", "C": "Stage 3", "D": "Stage 4", "E": "Stage 5"}.get(part, part)
        location = f"Photo {image_id} | {stage_label}"
        if detail:
            location += f" | {detail}"
        attempt = event.get("attempt")
        attempt_text = f" (attempt {attempt})" if attempt else ""

        if event_name in {"step_start", "step_retry", "retry"}:
            if event_name == "step_start":
                LOGGER.info("%s | started%s", location, attempt_text)
            else:
                message = event.get("message") or event.get("error") or event.get("error_type")
                LOGGER.warning("%s | retrying%s: %s", location, attempt_text, message or "transient error")
            return
        if event_name == "step_end":
            status = event.get("status", "finished")
            duration = event.get("duration_ms")
            duration_text = f" in {duration:.0f} ms" if isinstance(duration, (int, float)) else ""
            if status == "failed":
                message = event.get("message") or event.get("error") or event.get("category") or "unknown error"
                LOGGER.error("%s | failed%s: %s", location, duration_text, message)
            elif status == "skipped":
                LOGGER.info("%s | skipped: %s", location, event.get("reason", "checkpoint"))
            else:
                LOGGER.info("%s | %s%s", location, status, duration_text)


class PipelineEventLogger:
    """Write pipeline events to the run log and the matching part log."""

    def __init__(self, log_dir: str | Path, run_id: str) -> None:
        log_dir = Path(log_dir)
        self.run_id = run_id
        self.run_logger = JsonlEventLogger(log_dir / f"run_{run_id}.jsonl")
        self.part_loggers = {
            part: JsonlEventLogger(
                log_dir / f"part_{part.lower()}.jsonl", console=False
            )
            for part in "ABCD"
        }
        for part, logger in self.part_loggers.items():
            logger.write(
                {
                    "run_id": run_id,
                    "event": "part_log_start",
                    "part": part,
                }
            )

    def write(self, event: dict[str, Any]) -> None:
        self.run_logger.write(event)

        step = event.get("step")
        if isinstance(step, str):
            part = step.split("_", 1)[0]
            if part in self.part_loggers:
                self.part_loggers[part].write(event)

        if event.get("event") == "run_end":
            results = event.get("results")
            if not isinstance(results, list):
                return
            for result in results:
                if not isinstance(result, dict):
                    continue
                stage = result.get("stage")
                if isinstance(stage, str) and stage in self.part_loggers:
                    self.part_loggers[stage].write(
                        {
                            "run_id": event.get("run_id", self.run_id),
                            "image_id": result.get("image_id"),
                            "step": stage,
                            "event": "photo_result",
                            **result,
                        }
                    )


def configure_logging(level: int = logging.INFO) -> None:
    """Configure timestamped console logging for the command-line pipeline."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
