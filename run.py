from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.getenv("INGESTION_HOST", "127.0.0.1")
    port = int(os.getenv("INGESTION_PORT", "8000"))
    reload_enabled = os.getenv("INGESTION_RELOAD", "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    uvicorn.run(
        "app.asgi:app",
        host=host,
        port=port,
        reload=reload_enabled,
    )


if __name__ == "__main__":
    main()
