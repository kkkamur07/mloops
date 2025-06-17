from fastapi.testclient import TestClient
from main import app
client = TestClient(app)

def test_read_root():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "Welcome to the MNIST model inference API!"}
