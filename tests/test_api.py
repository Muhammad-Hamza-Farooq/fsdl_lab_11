from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_prediction():
    response = client.post("/predict", json=[1, 2, 3])
    assert response.status_code == 200
    assert response.json() == {"prediction": 6}


def test_invalid_input():
    response = client.post("/predict", json=["a", "b", "c"])
    assert response.status_code == 422


def test_empty_payload():
    response = client.post("/predict", json=[])
    assert response.status_code == 400
    assert response.json()["detail"] == "Empty payload"


def test_huge_request():
    huge_payload = list(range(10_001))
    response = client.post("/predict", json=huge_payload)
    assert response.status_code == 413


def test_concurrent_requests():
    def send_request():
        return client.post("/predict", json=[1, 2, 3, 4])

    with ThreadPoolExecutor(max_workers=10) as executor:
        responses = list(executor.map(lambda _: send_request(), range(20)))

    assert all(r.status_code == 200 for r in responses)
    assert all(r.json()["prediction"] == 10 for r in responses)
