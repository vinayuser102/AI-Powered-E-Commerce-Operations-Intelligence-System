from fastapi.testclient import TestClient

from api import app as app_module


class EmptyCollection:
    def query(self, **_kwargs):
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}


def test_health_endpoint(monkeypatch):
    monkeypatch.setattr(app_module, "MODEL_PATH", app_module.PROJECT_ROOT / "missing-model.pkl")
    with TestClient(app_module.app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_accepts_valid_features(monkeypatch):
    monkeypatch.setattr(app_module, "MODEL_PATH", app_module.PROJECT_ROOT / "missing-model.pkl")
    with TestClient(app_module.app) as client:
        response = client.post(
            "/predict/customer-123",
            json={"recency": 120, "frequency": 3, "monetary_value": 50.0, "refund_rate": 0.2},
        )
    assert response.status_code == 200
    assert response.json()["risk_level"] == "HIGH"


def test_predict_rejects_invalid_features(monkeypatch):
    monkeypatch.setattr(app_module, "MODEL_PATH", app_module.PROJECT_ROOT / "missing-model.pkl")
    with TestClient(app_module.app) as client:
        response = client.post(
            "/predict/customer-123",
            json={"recency": -1, "frequency": 0, "monetary_value": -1, "refund_rate": 2},
        )
    assert response.status_code == 422


def test_rag_returns_grounded_fallback_without_context(monkeypatch):
    monkeypatch.setattr(app_module, "collection", EmptyCollection())
    with TestClient(app_module.app) as client:
        response = client.post("/api/v1/query", json={"query": "What is the refund policy?"})
    assert response.status_code == 200
    assert response.json() == {
        "answer": "Policy details are unavailable.",
        "status": "no_context",
        "sources": [],
    }
