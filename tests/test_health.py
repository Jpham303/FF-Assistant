from fastapi.testclient import TestClient

from ffa.api.main import app


def test_health() -> None:
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_dagster_defs_load() -> None:
    from dagster_defs import defs

    assert defs is not None
