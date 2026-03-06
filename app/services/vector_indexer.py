from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from langchain_core.documents import Document

from app.core.config import IngestionSettings

if TYPE_CHECKING:
    from langchain_openai import OpenAIEmbeddings
    from pinecone.db_data.index import Index


@dataclass(frozen=True, slots=True)
class VectorUpsertResult:
    provider: str
    index_name: str | None
    namespace: str | None
    indexed_vectors: int


class VectorIndexer(Protocol):
    def upsert_documents(
        self,
        *,
        job_id: str,
        namespace: str | None,
        documents: list[Document],
    ) -> VectorUpsertResult:
        ...

    def delete_job(
        self,
        *,
        job_id: str,
        namespace: str | None,
        expected_chunk_count: int,
    ) -> None:
        ...


class NoOpVectorIndexer:
    def upsert_documents(
        self,
        *,
        job_id: str,
        namespace: str | None,
        documents: list[Document],
    ) -> VectorUpsertResult:
        return VectorUpsertResult(
            provider="none",
            index_name=None,
            namespace=namespace,
            indexed_vectors=0,
        )

    def delete_job(
        self,
        *,
        job_id: str,
        namespace: str | None,
        expected_chunk_count: int,
    ) -> None:
        return None


class PineconeVectorIndexer:
    def __init__(
        self,
        *,
        api_key: str,
        index_name: str,
        openai_api_key: str,
        embedding_model: str,
        embedding_dimensions: int | None,
        default_namespace: str | None,
    ) -> None:
        self._index_name = index_name
        self._default_namespace = default_namespace
        from pinecone import Pinecone

        self._client = Pinecone(api_key=api_key)
        self._index: Index = self._client.Index(name=index_name)
        self._embedding_dimensions = self._resolve_embedding_dimensions(
            configured_dimensions=embedding_dimensions,
            embedding_model=embedding_model,
        )
        from langchain_openai import OpenAIEmbeddings

        embedding_kwargs: dict[str, str | int] = {
            "api_key": openai_api_key,
            "model": embedding_model,
        }
        if self._embedding_dimensions is not None:
            embedding_kwargs["dimensions"] = self._embedding_dimensions

        self._embeddings: OpenAIEmbeddings = OpenAIEmbeddings(**embedding_kwargs)

    def upsert_documents(
        self,
        *,
        job_id: str,
        namespace: str | None,
        documents: list[Document],
    ) -> VectorUpsertResult:
        resolved_namespace = namespace or self._default_namespace
        if not documents:
            return VectorUpsertResult(
                provider="pinecone",
                index_name=self._index_name,
                namespace=resolved_namespace,
                indexed_vectors=0,
            )

        embeddings = self._embeddings.embed_documents(
            [document.page_content for document in documents]
        )
        vectors: list[dict[str, Any]] = []
        for offset, (document, embedding) in enumerate(zip(documents, embeddings, strict=False)):
            chunk_index = self._resolve_chunk_index(document, offset)
            vectors.append(
                {
                    "id": self._vector_id(job_id, chunk_index),
                    "values": embedding,
                    "metadata": self._build_metadata(job_id, document, chunk_index),
                }
            )

        self._index.upsert(
            vectors=vectors,
            namespace=resolved_namespace,
            show_progress=False,
        )
        return VectorUpsertResult(
            provider="pinecone",
            index_name=self._index_name,
            namespace=resolved_namespace,
            indexed_vectors=len(vectors),
        )

    def delete_job(
        self,
        *,
        job_id: str,
        namespace: str | None,
        expected_chunk_count: int,
    ) -> None:
        resolved_namespace = namespace or self._default_namespace
        if expected_chunk_count > 0:
            self._index.delete(
                ids=[self._vector_id(job_id, idx) for idx in range(expected_chunk_count)],
                namespace=resolved_namespace,
            )
            return

        self._index.delete(
            namespace=resolved_namespace,
            filter={"job_id": {"$eq": job_id}},
        )

    @staticmethod
    def _resolve_chunk_index(document: Document, fallback_index: int) -> int:
        raw_index = document.metadata.get("chunk_index")
        if isinstance(raw_index, int):
            return raw_index
        return fallback_index

    @staticmethod
    def _vector_id(job_id: str, chunk_index: int) -> str:
        return f"{job_id}-{chunk_index:06d}"

    def _build_metadata(
        self,
        job_id: str,
        document: Document,
        chunk_index: int,
    ) -> dict[str, str | int | float | bool]:
        metadata: dict[str, str | int | float | bool] = {
            "job_id": job_id,
            "chunk_index": chunk_index,
            "text": document.page_content,
        }
        for key, value in document.metadata.items():
            if value is None:
                continue
            safe_key = str(key)
            if safe_key in metadata:
                safe_key = f"meta_{safe_key}"
            metadata[safe_key] = self._normalize_metadata_value(value)
        return metadata

    @staticmethod
    def _normalize_metadata_value(value: Any) -> str | int | float | bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (str, int, float)):
            return value
        return json.dumps(value, default=str, sort_keys=True)

    def _resolve_embedding_dimensions(
        self,
        *,
        configured_dimensions: int | None,
        embedding_model: str,
    ) -> int | None:
        if configured_dimensions is not None:
            return configured_dimensions
        if not embedding_model.startswith("text-embedding-3"):
            return None

        description = self._client.describe_index(name=self._index_name)
        dimension = self._extract_index_dimension(description)
        if dimension is None:
            return None
        return dimension

    @staticmethod
    def _extract_index_dimension(description: Any) -> int | None:
        raw_dimension = getattr(description, "dimension", None)
        if raw_dimension is None and isinstance(description, dict):
            raw_dimension = description.get("dimension")
        if isinstance(raw_dimension, int) and raw_dimension > 0:
            return raw_dimension
        return None


def build_vector_indexer(settings: IngestionSettings) -> VectorIndexer:
    provided = [
        settings.pinecone_api_key,
        settings.pinecone_index_name,
        settings.openai_api_key,
    ]
    if not any(provided):
        return NoOpVectorIndexer()
    if not all(provided):
        raise RuntimeError(
            "Pinecone indexing requires PINECONE_API_KEY, PINECONE_INDEX_NAME, and OPENAI_API_KEY."
        )

    return PineconeVectorIndexer(
        api_key=settings.pinecone_api_key or "",
        index_name=settings.pinecone_index_name or "",
        openai_api_key=settings.openai_api_key or "",
        embedding_model=settings.openai_embedding_model,
        embedding_dimensions=settings.openai_embedding_dimensions,
        default_namespace=settings.pinecone_namespace,
    )
