from fastapi.testclient import TestClient

from ffa.api.main import app


def test_health() -> None:
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_dagster_defs_load() -> None:
    from dagster_defs import defs

    assert defs is not None


def test_dagster_instance_classes_importable() -> None:
    """Every class dagster.yaml names must be installed, or the containers crash at start."""
    import importlib
    from pathlib import Path

    import yaml

    cfg = yaml.safe_load(Path("dagster.yaml").read_text())
    modules = {"dagster_postgres"} if "postgres" in cfg.get("storage", {}) else set()
    modules |= {v["module"] for v in cfg.values() if isinstance(v, dict) and "module" in v}
    for m in modules:
        importlib.import_module(m)
