from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest
from services import (
    BOOKING_MAP,
    MONTH_ABBREVIATION_MAP,
    SEARCH_SPACE,
    VARIABLE_RENAME_MAP,
    VARS_TO_DROP,
    VARS_TO_IMPUTE,
    VARS_TO_OHE,
)
from services.predictor import make_prediction
from services.preprocessing import NoVacancyDataProcessing
from unittest.mock import patch, MagicMock, AsyncMock
from services import predictor
from main import app
from routers import predict
from config import __model_version__


@pytest.mark.asyncio
async def test_end_to_end_pipeline(booking_data):
    """Test end-to-end pipeline using MLflow mocks."""

    # Create async session mock based on AsyncPostgresDB config in postgres.py
    mock_session = AsyncMock()
    mock_sessionmaker = MagicMock()
    mock_sessionmaker.return_value.__aenter__.return_value = mock_session
    mock_sessionmaker.return_value.__aexit__.return_value = None

    # Mock AsyncPostgresDB.create_session()
    # Mock where gold_db is used, not where it is defined. Import system creates a local reference.
    with (
        patch.object(
            predictor.gold_db, "create_session", return_value=mock_sessionmaker
        ),
        patch.object(predictor, "load_pipeline_and_processor_from_mlflow") as mock_load,
    ):

        mock_pipeline = MagicMock()

        # Predictions much match booking_data length
        num_rows = len(booking_data)
        mock_pipeline.predict.return_value = np.array([0] * num_rows)
        mock_pipeline.predict_proba.return_value = np.array([[0.8, 0.2]] * num_rows)

        mock_processor = MagicMock()
        mock_processor.transform.return_value = (
            booking_data.drop(columns=["booking_status"]),
            None,
        )

        # Mock MLflow return the trained pipeline and processor
        mock_load.return_value = (mock_pipeline, mock_processor)

        result = await make_prediction(booking_data, already_processed=False)

        # Assert
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(booking_data)  # Should match input length
        assert "prediction" in result.columns
        assert "probability_not_canceled" in result.columns
        assert "probabilities_canceled" in result.columns
        assert "booking_id" in result.columns

        # Verify mocks were called
        mock_sessionmaker.assert_called_once()
        mock_session.merge.assert_called()
        mock_session.commit.assert_awaited_once()
        mock_load.assert_called_once()


@pytest.mark.asyncio
async def test_api_prediction_integration(booking_data):
    """
    Test the API endpoint integration with the prediction endpoint
    """
    # Create a test client for the FastAPI app (from main.py)
    client = TestClient(app)

    # Mock the prediction from where it is used, i.e., i.e., routers/predict.py
    with patch.object(predict, "make_prediction") as mock_make_prediction:
        mock_make_prediction.return_value = pd.DataFrame(
            {
                "booking_id": booking_data["Booking_ID"].tolist(),
                "prediction": [0] * len(booking_data),
                "probability_not_canceled": [0.8] * len(booking_data),
                "probabilities_canceled": [0.2] * len(booking_data),
            }
        )

        # Test API call
        payload = {"data": booking_data.to_dict(orient="records")}
        response = client.post("/predict", json=payload)

        # Assertions
        assert response.status_code == 200

        # Response return the likelihood of cancellation
        response_data = response.json()
        assert "predictions" in response_data
        assert "version" in response_data

        # More specific checks
        assert response_data["predictions"] == [0.2] * len(booking_data)
        assert response_data["version"] == __model_version__

        # Verify the API correctly processed the input
        mock_make_prediction.assert_called_once()

        # Check that the function was called with a DataFrame
        call_args = mock_make_prediction.call_args[0][0]
        assert isinstance(call_args, pd.DataFrame)
        assert len(call_args) == len(booking_data)
