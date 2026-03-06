from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import IngestionSettings
from app.main import create_app
from app.services.vector_indexer import VectorUpsertResult


class FakeVectorIndexer:
    def __init__(self) -> None:
        self.upserts: list[tuple[str, str | None, int]] = []
        self.deletes: list[tuple[str, str | None, int]] = []

    def upsert_documents(self, *, job_id: str, namespace: str | None, documents: list) -> VectorUpsertResult:
        indexed_vectors = len(documents)
        self.upserts.append((job_id, namespace, indexed_vectors))
        return VectorUpsertResult(
            provider="pinecone",
            index_name="unit-test-index",
            namespace=namespace,
            indexed_vectors=indexed_vectors,
        )

    def delete_job(
        self,
        *,
        job_id: str,
        namespace: str | None,
        expected_chunk_count: int,
    ) -> None:
        self.deletes.append((job_id, namespace, expected_chunk_count))


@pytest.fixture
def test_client(tmp_path: Path) -> TestClient:
    settings = IngestionSettings(
        data_dir=tmp_path / "data",
        upload_dir=tmp_path / "uploads",
        artifact_dir=tmp_path / "artifacts",
        sqlite_path=tmp_path / "status" / "jobs.db",
        processing_delay_seconds=0.25,
        delete_poll_interval_seconds=0.05,
        upload_chunk_size_bytes=1024,
        langchain_chunk_size=10,
        langchain_chunk_overlap=2,
    )
    app = create_app(settings=settings)
    with TestClient(app) as client:
        yield client


def _wait_for_terminal_state(
    client: TestClient,
    job_id: str,
    timeout_seconds: float = 5.0,
) -> dict[str, object]:
    start = time.time()
    while time.time() - start < timeout_seconds:
        response = client.get(f"/api/v1/ingestion/jobs/{job_id}")
        assert response.status_code == 200
        payload = response.json()
        if payload["completion_event"]:
            return payload
        time.sleep(0.05)
    raise AssertionError(f"Job {job_id} did not reach a terminal state in time")


def test_upload_completes_and_sets_completion_event(test_client: TestClient) -> None:
    response = test_client.post(
        "/api/v1/ingestion/uploads",
        files={"file": ("notes.txt", b"Enterprise ingestion baseline", "text/plain")},
        data={"namespace": "knowledge-team", "metadata_json": '{"owner":"backend"}'},
    )
    assert response.status_code == 202
    payload = response.json()

    final_status = _wait_for_terminal_state(test_client, payload["job_id"])
    assert final_status["status"] == "COMPLETED"
    assert final_status["completion_event"] is True
    assert final_status["result"]["preview"] == "Enterprise ingestion baseline"
    assert final_status["result"]["modality"] == "document"
    assert final_status["result"]["chunks_indexed"] >= 2
    assert final_status["result"]["langchain"]["splitter"] == "RecursiveCharacterTextSplitter"


def test_delete_mid_flight_cleans_up_and_marks_deleted(test_client: TestClient) -> None:
    response = test_client.post(
        "/api/v1/ingestion/uploads",
        files={"file": ("policy.txt", b"Delete me while ingesting", "text/plain")},
        data={"namespace": "knowledge-team"},
    )
    assert response.status_code == 202
    payload = response.json()

    delete_response = test_client.delete(f"/api/v1/ingestion/jobs/{payload['job_id']}")
    assert delete_response.status_code == 202

    final_status = _wait_for_terminal_state(test_client, payload["job_id"])
    assert final_status["status"] == "DELETED"
    assert final_status["completion_event"] is True

    upload_dir = Path(test_client.app.state.ingestion_service.settings.upload_dir)
    artifact_dir = Path(test_client.app.state.ingestion_service.settings.artifact_dir)
    assert list(upload_dir.glob("*")) == []
    assert list(artifact_dir.glob("*")) == []


def test_delete_after_completion_removes_completed_artifact(test_client: TestClient) -> None:
    response = test_client.post(
        "/api/v1/ingestion/uploads",
        files={"file": ("handbook.md", b"# Handbook", "text/markdown")},
    )
    assert response.status_code == 202
    payload = response.json()

    final_status = _wait_for_terminal_state(test_client, payload["job_id"])
    assert final_status["status"] == "COMPLETED"

    delete_response = test_client.delete(f"/api/v1/ingestion/jobs/{payload['job_id']}")
    assert delete_response.status_code == 202

    deleted_status = _wait_for_terminal_state(test_client, payload["job_id"])
    assert deleted_status["status"] == "DELETED"


def test_upload_can_write_to_configured_vector_indexer(tmp_path: Path) -> None:
    fake_vector_indexer = FakeVectorIndexer()
    with TestClient(
        create_app(
            settings=IngestionSettings(
                data_dir=tmp_path / "vector-data",
                upload_dir=tmp_path / "vector-data" / "uploads",
                artifact_dir=tmp_path / "vector-data" / "artifacts",
                sqlite_path=tmp_path / "vector-data" / "jobs.db",
                processing_delay_seconds=0.05,
                delete_poll_interval_seconds=0.01,
                upload_chunk_size_bytes=1024,
                langchain_chunk_size=8,
                langchain_chunk_overlap=0,
            ),
            vector_indexer=fake_vector_indexer,
        )
    ) as client:
        response = client.post(
            "/api/v1/ingestion/uploads",
            files={"file": ("vector.txt", b"Pinecone should receive these chunks", "text/plain")},
            data={"namespace": "vector-test"},
        )
        assert response.status_code == 202
        payload = response.json()

        final_status = _wait_for_terminal_state(client, payload["job_id"])
        assert final_status["status"] == "COMPLETED"
        assert final_status["result"]["vector_store"]["provider"] == "pinecone"
        assert final_status["result"]["vector_store"]["index_name"] == "unit-test-index"
        assert fake_vector_indexer.upserts

        delete_response = client.delete(f"/api/v1/ingestion/jobs/{payload['job_id']}")
        assert delete_response.status_code == 202
        deleted_status = _wait_for_terminal_state(client, payload["job_id"])
        assert deleted_status["status"] == "DELETED"
        assert fake_vector_indexer.deletes
