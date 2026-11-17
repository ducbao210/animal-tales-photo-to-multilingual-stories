"""Content fingerprints and validated per-photo stage checkpoints."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.contracts.schema import SCHEMA_VERSION
from src.utils.files import atomic_write_json, read_json


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_fingerprint(*parts: Any) -> str:
    encoded = json.dumps(parts, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return sha256_bytes(encoded.encode("utf-8"))


class RunState:
    def __init__(self, path: str | Path, image_id: str, input_sha256: str) -> None:
        self.path = Path(path)
        self.image_id = image_id
        self.input_sha256 = input_sha256
        self.data: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "image_id": image_id,
            "input_sha256": input_sha256,
            "stage": {},
            "errors": [],
        }
        try:
            stored = read_json(self.path)
            if (stored.get("schema_version") == SCHEMA_VERSION
                    and stored.get("image_id") == image_id
                    and stored.get("input_sha256") == input_sha256):
                self.data = stored
        except (OSError, ValueError, AttributeError):
            pass

    def is_complete(
        self, stage: str, fingerprint: str, root: Path,
        validator: Callable[[Path], bool] | None = None,
    ) -> bool:
        item = self.data["stage"].get(stage, {})
        if item.get("status") != "complete" or item.get("fingerprint") != fingerprint:
            return False
        paths = item.get("artifact_paths", [])
        if not isinstance(paths, list) or not paths:
            return False
        for raw in paths:
            candidate = Path(raw)
            if not candidate.is_absolute():
                candidate = root / candidate
            try:
                candidate.resolve().relative_to(root.resolve())
            except ValueError:
                return False
            if not candidate.is_file() or (validator is not None and not validator(candidate)):
                return False
        return True

    def complete(self, stage: str, fingerprint: str, artifacts: list[Path], root: Path) -> None:
        relative = []
        for path in artifacts:
            resolved = path.resolve()
            try:
                relative.append(resolved.relative_to(root.resolve()).as_posix())
            except ValueError as error:
                raise ValueError("Checkpoint artifacts must stay under the photo output folder") from error
            if not resolved.is_file():
                raise FileNotFoundError(f"Cannot checkpoint missing artifact: {resolved.name}")
        self.data["stage"][stage] = {
            "status": "complete", "fingerprint": fingerprint,
            "artifact_paths": relative,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        self.save()

    def failed(self, failure: dict[str, Any]) -> None:
        self.data["errors"].append(dict(failure))
        stage = failure.get("stage", "unknown")
        self.data["stage"][stage] = {"status": "failed", "error": dict(failure)}
        self.save()

    def save(self) -> None:
        atomic_write_json(self.path, self.data)
