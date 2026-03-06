from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class IngestionSettings:
    data_dir: Path
    upload_dir: Path
    artifact_dir: Path
    sqlite_path: Path
    processing_delay_seconds: float
    delete_poll_interval_seconds: float
    upload_chunk_size_bytes: int
    langchain_chunk_size: int
    langchain_chunk_overlap: int
    pinecone_api_key: str | None = None
    pinecone_index_name: str | None = None
    pinecone_namespace: str | None = None
    openai_api_key: str | None = None
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dimensions: int | None = None

    def ensure_paths(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)


def _load_local_env_file(path: Path) -> None:
    if not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue

        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ[key] = value


def load_settings() -> IngestionSettings:
    _load_local_env_file(Path(".env.local"))

    data_dir = Path(os.getenv("INGESTION_DATA_DIR", "./data/ingestion"))
    upload_dir = Path(os.getenv("INGESTION_UPLOAD_DIR", str(data_dir / "uploads")))
    artifact_dir = Path(os.getenv("INGESTION_ARTIFACT_DIR", str(data_dir / "artifacts")))
    sqlite_path = Path(os.getenv("INGESTION_SQLITE_PATH", str(data_dir / "jobs.db")))
    processing_delay_seconds = float(os.getenv("INGESTION_DELAY_SECONDS", "1.0"))
    delete_poll_interval_seconds = float(
        os.getenv("INGESTION_DELETE_POLL_SECONDS", "0.2")
    )
    upload_chunk_size_bytes = int(
        os.getenv("INGESTION_UPLOAD_CHUNK_BYTES", str(1024 * 1024))
    )
    langchain_chunk_size = int(os.getenv("INGESTION_LANGCHAIN_CHUNK_SIZE", "1000"))
    langchain_chunk_overlap = int(
        os.getenv("INGESTION_LANGCHAIN_CHUNK_OVERLAP", "150")
    )
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
    pinecone_namespace = os.getenv("PINECONE_NAMESPACE")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_embedding_model = os.getenv(
        "OPENAI_EMBEDDING_MODEL",
        "text-embedding-3-small",
    )
    raw_openai_embedding_dimensions = os.getenv("OPENAI_EMBEDDING_DIMENSIONS")
    openai_embedding_dimensions = None
    if raw_openai_embedding_dimensions:
        openai_embedding_dimensions = int(raw_openai_embedding_dimensions)

    return IngestionSettings(
        data_dir=data_dir,
        upload_dir=upload_dir,
        artifact_dir=artifact_dir,
        sqlite_path=sqlite_path,
        processing_delay_seconds=processing_delay_seconds,
        delete_poll_interval_seconds=delete_poll_interval_seconds,
        upload_chunk_size_bytes=upload_chunk_size_bytes,
        langchain_chunk_size=langchain_chunk_size,
        langchain_chunk_overlap=langchain_chunk_overlap,
        pinecone_api_key=pinecone_api_key,
        pinecone_index_name=pinecone_index_name,
        pinecone_namespace=pinecone_namespace,
        openai_api_key=openai_api_key,
        openai_embedding_model=openai_embedding_model,
        openai_embedding_dimensions=openai_embedding_dimensions,
    )
