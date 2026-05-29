from fastapi.testclient import TestClient

from secure_app import app

client = TestClient(app)


def test_huge_payload():
    response = client.post("/predict", json=list(range(1001)))
    assert response.status_code == 413


def test_invalid_json():
    response = client.post("/predict", content=b"{bad json", headers={"Content-Type": "application/json"})
    assert response.status_code == 422


def test_adversarial_input():
    response = client.post("/predict", json=[1e9, 1e9, 1e9])
    assert response.status_code == 422


def test_malformed_request():
    response = client.post("/predict", json={"not": "a list"})
    assert response.status_code == 422


def test_valid_request():
    response = client.post("/predict", json=[1, 2, 3])
    assert response.status_code == 200
