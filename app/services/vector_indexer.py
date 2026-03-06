from __future__ import annotations

import hashlib
import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from langchain_core.documents import Document

from app.core.config import IngestionSettings

if TYPE_CHECKING:
    from chromadb.api.models.Collection import Collection
    from langchain_openai import OpenAIEmbeddings


@dataclass(frozen=True, slots=True)
class VectorUpsertResult:
    provider: str
    index_name: str | None
    namespace: str | None
    indexed_vectors: int


class VectorIndexer(ABC):
    @abstractmethod
    def upsert_documents(
        self,
        *,
        job_id: str,
        namespace: str | None,
        documents: list[Document],
    ) -> VectorUpsertResult:
        raise NotImplementedError

    @abstractmethod
    def delete_job(
        self,
        *,
        job_id: str,
        namespace: str | None,
        expected_chunk_count: int,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def connectivity_status(self) -> dict[str, str]:
        raise NotImplementedError


class NoOpVectorIndexer(VectorIndexer):
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

    def connectivity_status(self) -> dict[str, str]:
        return {
            "status": "skipped",
            "provider": "none",
            "reason": "vector indexing disabled",
        }


class UnavailableVectorIndexer(VectorIndexer):
    def __init__(self, *, provider: str, reason: str) -> None:
        self._provider = provider
        self._reason = reason

    def upsert_documents(
        self,
        *,
        job_id: str,
        namespace: str | None,
        documents: list[Document],
    ) -> VectorUpsertResult:
        raise RuntimeError(
            f"Vector indexer '{self._provider}' unavailable: {self._reason}"
        )

    def delete_job(
        self,
        *,
        job_id: str,
        namespace: str | None,
        expected_chunk_count: int,
    ) -> None:
        return None

    def connectivity_status(self) -> dict[str, str]:
        return {
            "status": "down",
            "provider": self._provider,
            "error": self._reason,
        }


class TextEmbedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...


class OpenAITextEmbedder:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        dimensions: int | None,
    ) -> None:
        from langchain_openai import OpenAIEmbeddings

        kwargs: dict[str, str | int] = {"api_key": api_key, "model": model}
        if dimensions is not None:
            kwargs["dimensions"] = dimensions
        self._embedder: OpenAIEmbeddings = OpenAIEmbeddings(**kwargs)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.embed_documents(texts)


class HashTextEmbedder:
    def __init__(self, *, dimensions: int = 384) -> None:
        self._dimensions = max(64, dimensions)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_text(text) for text in texts]

    def _embed_text(self, text: str) -> list[float]:
        vector = [0.0] * self._dimensions
        tokens = text.split()
        if not tokens:
            tokens = ["_empty_"]

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], byteorder="big") % self._dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


class ChromaVectorIndexer(VectorIndexer):
    def __init__(
        self,
        *,
        persist_dir: str,
        collection_name: str,
        embedder: TextEmbedder,
        embedding_provider: str,
    ) -> None:
        import chromadb

        self._embedding_provider = embedding_provider
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection: Collection = self._client.get_or_create_collection(
            name=collection_name
        )
        self._collection_name = collection_name
        self._embedder = embedder

    def upsert_documents(
        self,
        *,
        job_id: str,
        namespace: str | None,
        documents: list[Document],
    ) -> VectorUpsertResult:
        if not documents:
            return VectorUpsertResult(
                provider="chroma",
                index_name=self._collection_name,
                namespace=namespace,
                indexed_vectors=0,
            )

        ids: list[str] = []
        metadatas: list[dict[str, str | int | float | bool]] = []
        texts: list[str] = []
        for offset, document in enumerate(documents):
            chunk_index = self._resolve_chunk_index(document, offset)
            ids.append(self._vector_id(job_id, chunk_index))
            metadatas.append(
                self._build_metadata(job_id=job_id, namespace=namespace, document=document, chunk_index=chunk_index)
            )
            texts.append(document.page_content)

        embeddings = self._embedder.embed_documents(texts)
        self._collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return VectorUpsertResult(
            provider="chroma",
            index_name=self._collection_name,
            namespace=namespace,
            indexed_vectors=len(ids),
        )

    def delete_job(
        self,
        *,
        job_id: str,
        namespace: str | None,
        expected_chunk_count: int,
    ) -> None:
        if expected_chunk_count > 0:
            self._collection.delete(
                ids=[self._vector_id(job_id, idx) for idx in range(expected_chunk_count)]
            )
            return

        self._collection.delete(where={"job_id": job_id})

    def connectivity_status(self) -> dict[str, str]:
        try:
            self._collection.count()
            return {
                "status": "up",
                "provider": "chroma",
                "collection": self._collection_name,
                "embedding_provider": self._embedding_provider,
            }
        except Exception as exc:
            return {
                "status": "down",
                "provider": "chroma",
                "collection": self._collection_name,
                "embedding_provider": self._embedding_provider,
                "error": str(exc),
            }

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
        *,
        job_id: str,
        namespace: str | None,
        document: Document,
        chunk_index: int,
    ) -> dict[str, str | int | float | bool]:
        metadata: dict[str, str | int | float | bool] = {
            "job_id": job_id,
            "chunk_index": chunk_index,
        }
        if namespace:
            metadata["namespace"] = namespace
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


def _build_embedder(settings: IngestionSettings) -> tuple[TextEmbedder, str]:
    if settings.openai_api_key:
        return (
            OpenAITextEmbedder(
                api_key=settings.openai_api_key,
                model=settings.openai_embedding_model,
                dimensions=settings.openai_embedding_dimensions,
            ),
            "openai",
        )
    local_dimensions = settings.openai_embedding_dimensions or 384
    return HashTextEmbedder(dimensions=local_dimensions), "hash-local"


def build_vector_indexer(settings: IngestionSettings) -> VectorIndexer:
    provider = settings.vector_provider
    if provider in {"none", "off", "disabled"}:
        return NoOpVectorIndexer()
    if provider != "chroma":
        return UnavailableVectorIndexer(
            provider=provider,
            reason=f"Unsupported provider '{provider}'. Supported providers: chroma, none",
        )

    try:
        embedder, embedding_provider = _build_embedder(settings)
        return ChromaVectorIndexer(
            persist_dir=str(settings.chroma_persist_dir),
            collection_name=settings.chroma_collection_name,
            embedder=embedder,
            embedding_provider=embedding_provider,
        )
    except Exception as exc:
        return UnavailableVectorIndexer(provider="chroma", reason=str(exc))
