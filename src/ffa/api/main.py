"""FastAPI app. Phase 0 exposes only a health check; endpoints arrive in Phase 1."""

from fastapi import FastAPI

from ffa import __version__

app = FastAPI(title="FF Assistant API", version=__version__)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}
