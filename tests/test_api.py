
import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert "model_loaded" in response.json()


def test_predict_valid_payload(client):
    response = client.post(
        "/predict",
        json={"features": [5.1, 3.5, 1.4, 0.2]},
    )

    assert response.status_code == 200

    body = response.json()

    assert body["prediction"] in {
        "setosa",
        "versicolor",
        "virginica",
    }

    probabilities = body["probabilities"]

    assert set(probabilities.keys()) == {"0", "1", "2"}
    assert abs(sum(probabilities.values()) - 1.0) < 0.02


def test_rejects_wrong_feature_count(client):
    response = client.post(
        "/predict",
        json={"features": [1.0, 2.0]},
    )

    assert response.status_code == 422


def test_rejects_extra_fields(client):
    response = client.post(
        "/predict",
        json={
            "features": [5.1, 3.5, 1.4, 0.2],
            "unexpected": True,
        },
    )

    assert response.status_code == 422


def test_rejects_negative_values(client):
    response = client.post(
        "/predict",
        json={"features": [-1, 3.5, 1.4, 0.2]},
    )

    assert response.status_code == 422
