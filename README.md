# ingestion-pipeline

FastAPI ingestion baseline for an enterprise knowledge platform.

This service is a cleaner starting point for the old GC `runner-gpt` ingestion flow:

- accepts file uploads from the UI
- writes a durable ingestion job record to SQLite
- processes the file asynchronously
- exposes a polling endpoint for job status
- honors mid-flight delete requests and cleans up local artifacts before setting the terminal completion event
- builds LangChain `Document` objects and chunks them with `RecursiveCharacterTextSplitter`
- stores vectors in local persistent Chroma DB by default
- keeps vector provider pluggable via `INGESTION_VECTOR_PROVIDER` (`chroma` now, easy to add more later)

The first processor is intentionally simple, but it now uses LangChain primitives as the baseline. It converts uploads into LangChain documents, performs chunking, classifies a rough modality, and writes a local artifact JSON. That gives you the right seam to plug in embeddings, vector stores, and retrieval later without changing the API contract.

## API

- `GET /api/v1/ingestion/jobs`
- `POST /api/v1/ingestion/uploads`
- `GET /api/v1/ingestion/jobs/{job_id}`
- `DELETE /api/v1/ingestion/jobs/{job_id}`
- `GET /healthz`

## Run

```bash
pip install -r requirements-dev.txt
uvicorn app.asgi:app --reload
```

For PyCharm "Run file", you can also use:

```bash
python run.py
```

The app auto-loads local secrets from `.env.local` if that file exists.

## Test

```bash
pytest
```

## Quality Checks

```bash
ruff check app tests
mypy app
pytest
```

You can also install the package itself with pip:

```bash
pip install -e .
```

Optional vector indexing env vars:

```bash
INGESTION_VECTOR_PROVIDER=chroma
INGESTION_CHROMA_PERSIST_DIR=./data/ingestion/chroma
INGESTION_CHROMA_COLLECTION=ingestion_chunks

# Optional OpenAI embeddings (if omitted, local hash embeddings are used)
OPENAI_API_KEY=your-openai-api-key
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_EMBEDDING_DIMENSIONS=1536
```

Put real values in `.env.local`, not in tracked docs or source files.
