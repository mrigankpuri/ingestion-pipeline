from __future__ import annotations

import hashlib
import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from langchain_core.documents import Document
from pydantic import SecretStr

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
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def connectivity_status(self) -> dict[str, str]:
        raise NotImplementedError


class StaticVectorIndexer(VectorIndexer):
    def __init__(
        self,
        *,
        provider: str,
        status: str,
        reason: str,
        fail_on_upsert: bool,
    ) -> None:
        self._provider = provider
        self._status = status
        self._reason = reason
        self._fail_on_upsert = fail_on_upsert

    def upsert_documents(
        self,
        *,
        job_id: str,
        namespace: str | None,
        documents: list[Document],
    ) -> VectorUpsertResult:
        if self._fail_on_upsert:
            raise RuntimeError(f"Vector indexer '{self._provider}' unavailable: {self._reason}")
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
    ) -> None:
        return None

    def connectivity_status(self) -> dict[str, str]:
        if self._status == "skipped":
            return {
                "status": "skipped",
                "provider": self._provider,
                "reason": self._reason,
            }
        return {
            "status": "down",
            "provider": self._provider,
            "error": self._reason,
        }


class ChromaVectorIndexer(VectorIndexer):
    def __init__(self, *, settings: IngestionSettings) -> None:
        import chromadb

        self._collection_name = settings.chroma_collection_name
        self._client = chromadb.PersistentClient(path=str(settings.chroma_persist_dir))
        self._collection: Collection = self._client.get_or_create_collection(
            name=self._collection_name
        )
        self._openai_embeddings: OpenAIEmbeddings | None = None
        self._local_dimensions = settings.openai_embedding_dimensions or 384
        self._embedding_provider = "hash-local"

        if settings.openai_api_key:
            from langchain_openai import OpenAIEmbeddings

            self._openai_embeddings = OpenAIEmbeddings(
                api_key=SecretStr(settings.openai_api_key),
                model=settings.openai_embedding_model,
                dimensions=settings.openai_embedding_dimensions,
            )
            self._embedding_provider = "openai"

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
        texts: list[str] = []
        metadatas: list[dict[str, str | int | float | bool]] = []
        for offset, document in enumerate(documents):
            chunk_index = self._resolve_chunk_index(document, offset)
            ids.append(f"{job_id}-{chunk_index:06d}")
            texts.append(document.page_content)
            metadatas.append(self._build_metadata(job_id, chunk_index, namespace, document))

        embeddings = self._embed_documents(texts)
        self._collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=cast(Any, embeddings),
            metadatas=cast(Any, metadatas),
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
    ) -> None:
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

    def _embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self._openai_embeddings is not None:
            return self._openai_embeddings.embed_documents(texts)
        return [self._hash_embed_text(text) for text in texts]

    def _hash_embed_text(self, text: str) -> list[float]:
        vector = [0.0] * max(64, self._local_dimensions)
        tokens = text.split() or ["_empty_"]
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], byteorder="big") % len(vector)
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    @staticmethod
    def _resolve_chunk_index(document: Document, fallback_index: int) -> int:
        raw_index = document.metadata.get("chunk_index")
        if isinstance(raw_index, int):
            return raw_index
        return fallback_index

    def _build_metadata(
        self,
        job_id: str,
        chunk_index: int,
        namespace: str | None,
        document: Document,
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


def build_vector_indexer(settings: IngestionSettings) -> VectorIndexer:
    provider = settings.vector_provider
    if provider in {"none", "off", "disabled"}:
        return StaticVectorIndexer(
            provider="none",
            status="skipped",
            reason="vector indexing disabled",
            fail_on_upsert=False,
        )
    if provider != "chroma":
        return StaticVectorIndexer(
            provider=provider,
            status="down",
            reason=f"Unsupported provider '{provider}'. Supported providers: chroma, none",
            fail_on_upsert=True,
        )

    try:
        return ChromaVectorIndexer(settings=settings)
    except Exception as exc:
        return StaticVectorIndexer(
            provider="chroma",
            status="down",
            reason=str(exc),
            fail_on_upsert=True,
        )
