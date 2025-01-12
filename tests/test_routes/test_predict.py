from fastapi.testclient import TestClient
from app.main import app
import pandas as pd
import pytest


client = TestClient(app)


def test_predict_endpoint(booking_data):
    # Arrange
    sample_obs = pd.DataFrame(booking_data.iloc[[0]].copy())
    payload = {"data": sample_obs.to_dict(orient="records")}

    # Act
    response = client.post("/predict", json=payload)
    response_json = response.json()

    # Assert
    assert response.status_code == 200, f"Failed with status code {response.status_code}"
    assert "predictions" in response_json, "Response does not contain 'predictions'"
    assert "version" in response_json, "Response does not contain 'version'"
    assert isinstance(response_json["predictions"], list), "'predictions' should be a list"
