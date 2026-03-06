from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, cast

from fastapi import FastAPI

from app.api.v1.ingestion import router as ingestion_router
from app.core.config import IngestionSettings, load_settings
from app.repository.ingestion_jobs import IngestionRepository
from app.services.ingestion import IngestionService
from app.services.processors.langchain_processor import (
    IngestionProcessor,
    LangChainIngestionProcessor,
)
from app.services.storage import LocalFileStore
from app.services.vector_indexer import VectorIndexer, build_vector_indexer


def create_app(
    settings: IngestionSettings | None = None,
    processor: IngestionProcessor | None = None,
    vector_indexer: VectorIndexer | None = None,
) -> FastAPI:
    resolved_settings = settings or load_settings()
    resolved_settings.ensure_paths()

    repository = IngestionRepository(resolved_settings.sqlite_path)
    file_store = LocalFileStore(resolved_settings)
    resolved_vector_indexer = vector_indexer or build_vector_indexer(resolved_settings)
    service = IngestionService(
        settings=resolved_settings,
        repository=repository,
        file_store=file_store,
        vector_indexer=resolved_vector_indexer,
        processor=processor
        or LangChainIngestionProcessor(
            chunk_size=resolved_settings.langchain_chunk_size,
            chunk_overlap=resolved_settings.langchain_chunk_overlap,
        ),
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
            await service.shutdown()
            repository.close()

    app = FastAPI(
        title="Enterprise Knowledge Ingestion Service",
        version="0.1.0",
        description=(
            "Modular ingestion baseline with upload, asynchronous processing, "
            "status polling, and delete-aware cleanup."
        ),
        lifespan=lifespan,
    )
    # Cast through Any to avoid IDE false positives on dynamic FastAPI state attributes.
    app_state = cast(Any, app).state
    setattr(app_state, "ingestion_repository", repository)
    setattr(app_state, "ingestion_service", service)
    setattr(app_state, "ingestion_vector_indexer", resolved_vector_indexer)
    app.include_router(ingestion_router)

    @app.get("/healthz")
    async def healthz() -> dict[str, object]:
        components = {
            "sqlite": repository.connectivity_status(),
            "storage": file_store.connectivity_status(),
            "vector_store": resolved_vector_indexer.connectivity_status(),
        }
        is_ok = all(
            component.get("status") in {"up", "skipped"}
            for component in components.values()
        )
        return {"status": "ok" if is_ok else "degraded", "components": components}

    return app
