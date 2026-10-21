import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

FX = json.loads((Path(__file__).parent / "fixture.json").read_text())


@pytest.fixture(params=["logreg", "xgb", "mlp"])
def client(request, monkeypatch):
    monkeypatch.setenv("MODEL_NAME", request.param)
    from inference.app import create_app
    with TestClient(create_app()) as c:
        yield c, request.param


def test_health_and_ready(client):
    c, name = client
    assert c.get("/healthz").status_code == 200
    assert c.get("/readyz").json()["model"] == name


def test_known_answers(client):
    c, name = client
    r = c.post("/score", json={"instances": FX["instances"]})
    assert r.status_code == 200
    got = r.json()["scores"]
    assert len(got) == len(FX["instances"])
    for g, e in zip(got, FX["expected"][name]):
        assert abs(g - e) < 1e-5
        assert 0.0 <= g <= 1.0


def test_rejects_wrong_feature_count(client):
    c, _ = client
    assert c.post("/score", json={"instances": [[0.0] * 28]}).status_code == 422


def test_rejects_empty_batch(client):
    c, _ = client
    assert c.post("/score", json={"instances": []}).status_code == 422


def test_metrics_endpoint(client):
    c, _ = client
    c.post("/score", json={"instances": FX["instances"][:1]})
    assert "inference_requests_total" in c.get("/metrics").text
