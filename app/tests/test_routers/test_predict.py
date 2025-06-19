from unittest.mock import patch
import pandas as pd
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_predict_endpoint(booking_data, mock_mlflow_client):
    """
    Test prediction endpoint with MLflow mocking.
    """
    # Arrange
    sample_obs = pd.DataFrame(booking_data.iloc[[0]].copy())
    payload = {"data": sample_obs.to_dict(orient="records")}

    # Mock the make_prediction function
    with patch("routers.predict.make_prediction") as mock_predict:
        mock_predict.return_value = {
            "booking_id": ["INN00001"],
            "prediction": [0],
            "probability_not_canceled": [0.8],
            "probabilities_canceled": [0.2],
        }

        # Act
        response = client.post("/predict", json=payload)
        response_json = response.json()

    # Assert
    assert (
        response.status_code == 200
    ), f"Failed with status code {response.status_code}"
    assert "predictions" in response_json, "Response does not contain 'predictions'"
    assert "version" in response_json, "Response does not contain 'version'"
