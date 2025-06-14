from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from config import MLFLOW_TRACKING_URI
from services import MLFLOW_EXPERIMENT_NAME, MLFLOW_PROCESSOR_PATH
from services.predictor import load_pipeline_and_processor_from_mlflow, make_prediction
from services import predictor


@patch("services.predictor.joblib.load")
@patch("services.predictor.mlflow.artifacts.download_artifacts")
@patch("services.predictor.mlflow.sklearn.load_model")
@patch("services.predictor.mlflow.MlflowClient")
@patch("services.predictor.mlflow.set_tracking_uri")
def test_load_pipeline_and_processor_from_mlflow_success(
    mock_set_tracking_uri,
    mock_client_class,
    mock_load_model,
    mock_download_artifacts,
    mock_joblib_load,
):
    # Arrange
    fake_pipeline = MagicMock(name="pipeline")
    fake_processor = MagicMock(name="processor")

    mock_client = MagicMock()
    mock_client.get_latest_versions.return_value = [MagicMock(run_id="fake_run_id")]
    mock_client_class.return_value = mock_client

    mock_load_model.return_value = fake_pipeline
    mock_download_artifacts.return_value = "tmp/processor.joblib"
    mock_joblib_load.return_value = fake_processor

    # Act
    pipeline, processor = load_pipeline_and_processor_from_mlflow(stage="Production")

    # Assert
    mock_set_tracking_uri.assert_called_once_with(MLFLOW_TRACKING_URI)
    mock_client.get_latest_versions.assert_called_once_with(
        MLFLOW_EXPERIMENT_NAME, stages=["Production"]
    )
    mock_load_model.assert_called_once_with(
        model_uri=f"models:/{MLFLOW_EXPERIMENT_NAME}/Production"
    )
    mock_download_artifacts.assert_called_once_with(
        run_id="fake_run_id", artifact_path=MLFLOW_PROCESSOR_PATH
    )
    mock_joblib_load.assert_called_once_with("tmp/processor.joblib")

    assert pipeline == fake_pipeline
    assert processor == fake_processor


@patch("services.predictor.mlflow.MlflowClient")
@patch("services.predictor.mlflow.set_tracking_uri")
def test_load_pipeline_and_processor_from_mlflow_no_versions(
    mock_set_tracking_uri, mock_client_class
):
    # Arrange
    mock_client = MagicMock()
    mock_client.get_latest_versions.return_value = []
    mock_client_class.return_value = mock_client

    # Act & Assert
    with pytest.raises(
        RuntimeError, match="No model version found for stage 'Staging'"
    ):
        load_pipeline_and_processor_from_mlflow(stage="Staging")

    # Ensure coverage of set_tracking_uri side effect with correct URI
    mock_set_tracking_uri.assert_called_once_with(MLFLOW_TRACKING_URI)


@pytest.mark.asyncio
@patch("services.predictor.load_pipeline_and_processor_from_mlflow")
async def test_make_prediction_integration_mock(
    mock_load_pipeline, booking_data, mock_pipeline, mock_processor
):
    # mock_lock_pipeline is automatically injected by @patch
    # Arrange
    # Use only a slice to keep the test lightweight
    df = booking_data[
        ["Booking_ID", "number of adults", "number of children", "booking status"]
    ].copy()
    df.rename(
        columns={"Booking_ID": "booking_id", "booking status": "booking_status"},
        inplace=True,
    )

    # Mock pipeline & processor behavior
    mock_pipeline.predict.return_value = np.array([1] * len(df))
    mock_pipeline.predict_proba.return_value = np.array([[0.8, 0.2]] * len(df))
    mock_processor.transform.return_value = (
        np.array([[0.1, 0.2, 0.3]] * len(df)),
        np.array([1] * len(df)),
    )

    mock_load_pipeline.return_value = (mock_pipeline, mock_processor)

    # Patch gold_db.create_session() to return a sessionmaker that returns an AsyncSession
    mock_session = AsyncMock()
    mock_sessionmaker = MagicMock(return_value=mock_session)
    mock_session.__aenter__.return_value = mock_session

    # Always patch the import path in the module that's being tested, not the original module
    with patch.object(
        predictor.gold_db, "create_session", return_value=mock_sessionmaker
    ):
        # Act
        result = await make_prediction(df)

    # Assert
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(df)
    assert result["booking_id"].iloc[0] == df["booking_id"].iloc[0]
    mock_session.merge.assert_called()  # Confirm insert logic ran
    mock_session.commit.assert_awaited()
    assert set(result.columns) >= {
        "booking_id",
        "prediction",
        "probability_not_canceled",
        "probabilities_canceled",
    }  # >= operator checks if the set on the left is a superset of the set on the right
