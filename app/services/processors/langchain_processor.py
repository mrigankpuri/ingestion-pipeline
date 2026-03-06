from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


DOCUMENT_EXTENSIONS = {
    ".csv",
    ".doc",
    ".docx",
    ".md",
    ".pdf",
    ".ppt",
    ".pptx",
    ".rtf",
    ".txt",
    ".xls",
    ".xlsx",
}
IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".svg", ".tiff", ".webp"}
AUDIO_EXTENSIONS = {".aac", ".flac", ".m4a", ".mp3", ".wav"}
VIDEO_EXTENSIONS = {".avi", ".m4v", ".mov", ".mp4", ".mkv", ".webm"}


@dataclass(frozen=True, slots=True)
class ProcessingOutcome:
    documents: list[Document]
    result: dict[str, Any]
    artifact_payload: dict[str, Any]


class IngestionProcessor(Protocol):
    def process(
        self,
        *,
        job_id: str,
        file_path: Path,
        namespace: str | None,
        metadata: dict[str, Any],
        content_type: str | None,
    ) -> ProcessingOutcome:
        ...


class LangChainIngestionProcessor:
    def __init__(self, *, chunk_size: int = 1000, chunk_overlap: int = 150) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def process(
        self,
        *,
        job_id: str,
        file_path: Path,
        namespace: str | None,
        metadata: dict[str, Any],
        content_type: str | None,
    ) -> ProcessingOutcome:
        raw = file_path.read_bytes()
        modality = self._infer_modality(file_path=file_path, content_type=content_type)
        source_documents = self._build_source_documents(
            file_path=file_path,
            raw=raw,
            modality=modality,
            metadata=metadata,
            content_type=content_type,
        )
        chunk_documents = self._split_documents(
            documents=source_documents,
            modality=modality,
        )
        preview = self._build_preview(chunk_documents)
        result = {
            "preview": preview,
            "bytes_processed": len(raw),
            "chunks_indexed": len(chunk_documents),
            "modality": modality,
            "langchain": {
                "document_count": len(source_documents),
                "chunk_count": len(chunk_documents),
                "splitter": "RecursiveCharacterTextSplitter",
            },
        }
        artifact_payload = {
            "job_id": job_id,
            "namespace": namespace,
            "metadata": metadata,
            "content_type": content_type,
            "source_file": file_path.name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "result": result,
            "langchain_documents": self._serialize_documents(chunk_documents),
        }
        return ProcessingOutcome(
            documents=chunk_documents,
            result=result,
            artifact_payload=artifact_payload,
        )

    def _build_source_documents(
        self,
        *,
        file_path: Path,
        raw: bytes,
        modality: str,
        metadata: dict[str, Any],
        content_type: str | None,
    ) -> list[Document]:
        page_content = self._coerce_page_content(raw)
        document_metadata = {
            "source": file_path.name,
            "source_path": str(file_path),
            "modality": modality,
        }
        if content_type:
            document_metadata["content_type"] = content_type
        document_metadata.update(metadata)

        return [Document(page_content=page_content, metadata=document_metadata)]

    def _split_documents(
        self,
        *,
        documents: list[Document],
        modality: str,
    ) -> list[Document]:
        if not documents:
            return []
        if modality != "document":
            return self._annotate_chunks(documents)

        chunks = self._splitter.split_documents(documents)
        if not chunks:
            chunks = documents
        return self._annotate_chunks(chunks)

    @staticmethod
    def _annotate_chunks(documents: list[Document]) -> list[Document]:
        annotated_documents: list[Document] = []
        for index, document in enumerate(documents):
            chunk_metadata = dict(document.metadata)
            chunk_metadata["chunk_index"] = index
            annotated_documents.append(
                Document(page_content=document.page_content, metadata=chunk_metadata)
            )
        return annotated_documents

    @staticmethod
    def _coerce_page_content(raw: bytes) -> str:
        if not raw:
            return ""
        try:
            decoded = raw.decode("utf-8")
        except UnicodeDecodeError:
            return f"[binary file: {len(raw)} bytes]"
        return " ".join(decoded.split())

    @staticmethod
    def _build_preview(documents: list[Document]) -> str:
        if not documents:
            return ""
        combined_text = " ".join(document.page_content for document in documents if document.page_content)
        return combined_text[:3000]

    @staticmethod
    def _serialize_documents(documents: list[Document]) -> list[dict[str, Any]]:
        serialized_documents: list[dict[str, Any]] = []
        for document in documents:
            serialized_documents.append(
                {
                    "page_content_preview": document.page_content[:500],
                    "character_count": len(document.page_content),
                    "metadata": document.metadata,
                }
            )
        return serialized_documents

    @staticmethod
    def _infer_modality(*, file_path: Path, content_type: str | None) -> str:
        if content_type:
            major_type = content_type.split("/", 1)[0].lower()
            if major_type == "image":
                return "image"
            if major_type == "audio":
                return "audio"
            if major_type == "video":
                return "video"
            if major_type in {"application", "text"}:
                return "document"

        suffix = file_path.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            return "image"
        if suffix in AUDIO_EXTENSIONS:
            return "audio"
        if suffix in VIDEO_EXTENSIONS:
            return "video"
        if suffix in DOCUMENT_EXTENSIONS:
            return "document"
        return "binary"
