from __future__ import annotations

from app.core.config import load_settings


def test_load_settings_reads_env_local_without_overriding_shell_env(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    for key in (
        "INGESTION_VECTOR_PROVIDER",
        "INGESTION_CHROMA_COLLECTION",
        "OPENAI_API_KEY",
        "OPENAI_EMBEDDING_DIMENSIONS",
    ):
        monkeypatch.delenv(key, raising=False)

    (tmp_path / ".env.local").write_text(
        "\n".join(
            [
                "INGESTION_VECTOR_PROVIDER=chroma",
                "OPENAI_API_KEY=file-openai-key",
                "OPENAI_EMBEDDING_DIMENSIONS=1024",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("INGESTION_CHROMA_COLLECTION", "shell-collection")

    settings = load_settings()

    assert settings.vector_provider == "chroma"
    assert settings.chroma_collection_name == "shell-collection"
    assert settings.openai_api_key == "file-openai-key"
    assert settings.openai_embedding_dimensions == 1024
