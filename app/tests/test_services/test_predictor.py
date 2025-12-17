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
        ["Booking_ID", "number_of_adults", "number_of_children", "booking_status"]
    ].copy()

    # Configure mocks
    mock_load_pipeline.return_value = (mock_pipeline, mock_processor)

    # Patch gold_db.create_session() to return a sessionmaker that returns an AsyncSession
    mock_session = AsyncMock()
    mock_sessionmaker = MagicMock(return_value=mock_session)
    mock_session.__aenter__.return_value = mock_session

    # Always patch the import path in the module that's being tested, not the original module
    # Mock where the object is used, not where it is defined
    with patch.object(
        predictor.gold_db, "create_session", return_value=mock_sessionmaker
    ):
        # Mock the processor
        processed_df = df.drop(columns=["booking_status"]).rename(
            columns={"Booking_ID": "booking_id"}
        )
        mock_processor.transform.return_value = (processed_df, None)

        # Mock the pipeline
        num_rows = len(df)
        mock_pipeline.predict.return_value = np.array([0] * num_rows)
        mock_pipeline.predict_proba.return_value = np.array([[0.8, 0.2]] * num_rows)

        # Act
        result = await make_prediction(df, already_processed=False)

    # Assert
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(df)
    assert result["booking_id"].iloc[0] == df["Booking_ID"].iloc[0]
    mock_session.merge.assert_called()  # Confirm insert logic ran
    mock_session.commit.assert_awaited()
    assert set(result.columns) >= {
        "booking_id",
        "prediction",
        "probability_not_canceled",
        "probabilities_canceled",
    }  # >= operator checks if the set on the left is a superset of the set on the right


@pytest.mark.asyncio
@patch("services.predictor.load_pipeline_and_processor_from_mlflow")
@patch.object(predictor.gold_db, "create_session")
async def test_make_prediction_frontend_payload_column_mismatch(
    mock_create_session, mock_load_pipeline, frontend_booking_payload, sample_processor
):
    """
    Integration test: verifies frontend payload survives preprocessing
    without column mismatch errors. Uses REAL processor to catch bugs.
    """
    # Arrange
    df = pd.DataFrame([frontend_booking_payload])

    mock_pipeline = MagicMock()
    mock_pipeline.predict.return_value = np.array([0])
    mock_pipeline.predict_proba.return_value = np.array([[0.7, 0.3]])

    # Use real processor, and mock pipeline
    mock_load_pipeline.return_value = (mock_pipeline, sample_processor)

    # Mock DB session
    mock_session = AsyncMock()
    mock_sessionmaker = MagicMock(return_value=mock_session)
    mock_session.__aenter__.return_value = mock_session
    mock_create_session.return_value = mock_sessionmaker

    # Act
    result = await make_prediction(df, already_processed=False)

    # Assert
    assert result is not None
    assert len(result) == 1

    # Verify pipeline received data WITHOUT booking_id and booking_status
    call_args = mock_pipeline.predict_proba.call_args[0][0]
    assert "booking_id" not in call_args.columns
    assert "booking_status" not in call_args.columns
