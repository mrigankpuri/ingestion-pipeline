from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from fastapi import UploadFile

from app.core.config import IngestionSettings


class LocalFileStore:
    def __init__(self, settings: IngestionSettings) -> None:
        self._settings = settings

    async def persist_upload(self, upload: UploadFile, stored_name: str) -> Path:
        destination = self._settings.upload_dir / stored_name
        destination.parent.mkdir(parents=True, exist_ok=True)

        with destination.open("wb") as output_file:
            while True:
                chunk = await upload.read(self._settings.upload_chunk_size_bytes)
                if not chunk:
                    break
                output_file.write(chunk)

        await upload.close()
        return destination

    def write_artifact(self, job_id: str, payload: dict[str, Any]) -> Path:
        destination = self._settings.artifact_dir / f"{job_id}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return destination

    def cleanup_paths(self, paths: list[str | Path | None]) -> None:
        seen: set[Path] = set()
        for raw_path in paths:
            if raw_path in (None, ""):
                continue
            file_path = Path(raw_path)
            if file_path in seen:
                continue
            seen.add(file_path)
            if file_path.exists():
                file_path.unlink()

    def connectivity_status(self) -> dict[str, str]:
        try:
            self._settings.upload_dir.mkdir(parents=True, exist_ok=True)
            self._settings.artifact_dir.mkdir(parents=True, exist_ok=True)
            probe_file = self._settings.artifact_dir / f".health-{uuid.uuid4().hex}"
            probe_file.write_text("ok", encoding="utf-8")
            probe_file.unlink(missing_ok=True)
            return {"status": "up"}
        except Exception as exc:
            return {"status": "down", "error": str(exc)}
