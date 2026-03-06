from __future__ import annotations

from app.core.config import load_settings


def test_load_settings_reads_env_local_without_overriding_shell_env(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    for key in (
        "PINECONE_API_KEY",
        "PINECONE_INDEX_NAME",
        "OPENAI_API_KEY",
        "OPENAI_EMBEDDING_DIMENSIONS",
    ):
        monkeypatch.delenv(key, raising=False)

    (tmp_path / ".env.local").write_text(
        "\n".join(
            [
                "PINECONE_API_KEY=file-pinecone-key",
                "OPENAI_API_KEY=file-openai-key",
                "OPENAI_EMBEDDING_DIMENSIONS=1024",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("PINECONE_INDEX_NAME", "shell-index")

    settings = load_settings()

    assert settings.pinecone_api_key == "file-pinecone-key"
    assert settings.pinecone_index_name == "shell-index"
    assert settings.openai_api_key == "file-openai-key"
    assert settings.openai_embedding_dimensions == 1024
