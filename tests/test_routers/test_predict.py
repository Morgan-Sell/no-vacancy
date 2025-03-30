import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_predict_endpoint(booking_data, trained_pipeline_and_processor):
    """
    Pipeline file must exist at app/models/no_vacancy_pipeline.pkl to run this test.
    """
    _, _, pm = trained_pipeline_and_processor
    
    # Arrange
    sample_obs = pd.DataFrame(booking_data.iloc[[0]].copy())
    payload = {"data": sample_obs.to_dict(orient="records")}

    # Act
    response = client.post("/predict", json=payload)
    response_json = response.json()

    # Assert
    assert (
        response.status_code == 200
    ), f"Failed with status code {response.status_code}"
    assert "predictions" in response_json, "Response does not contain 'predictions'"
    assert "version" in response_json, "Response does not contain 'version'"
    assert isinstance(
        response_json["predictions"], list
    ), "'predictions' should be a list"
