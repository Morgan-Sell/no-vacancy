from unittest.mock import patch
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_predict_endpoint(booking_data, trained_pipeline_and_processor):
    """
    Pipeline file must exist at app/models/no_vacancy_pipeline.pkl to run this test.
    """
    pipe, processor, _ = trained_pipeline_and_processor

    # Arrange: Create a sample observation for prediction
    sample_obs = booking_data.iloc[[0]].drop(columns=["booking status"])
    payload = {"data": sample_obs.to_dict(orient="records")}

    # Patch PipelineManagement.load_pipeline to return trained objects instead of reading from file
    with patch(
        "app.services.pipeline_management.PipelineManagement.load_pipeline",
        return_value=(pipe, processor),
    ) as mock_load_pipeline:
        # Act: Make a prediction request to the endpoint
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
