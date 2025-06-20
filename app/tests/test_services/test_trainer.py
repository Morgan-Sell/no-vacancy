import shutil
import tempfile
import time
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
import mlflow
import numpy as np
import pandas as pd
import joblib
import pytest
from app.services import DEPENDENT_VAR_NAME
from schemas.bronze import RawData
from services.pipeline import NoVacancyPipeline
from services.trainer import (
    build_pipeline,
    evaluate_model,
    load_raw_data,
    preprocess_data,
    save_to_silver_db,
    train_pipeline,
)
from services import trainer


@pytest.mark.asyncio
async def test_load_raw_data_from_bronze(mocker, booking_data):
    # Arrange: Convert booking_data to list of mocked RawData objects
    mock_records = []

    # Iterate through dataframe rows
    for _, row in booking_data.iterrows():
        record = RawData()

        for col, val in row.items():
            # Convert column names to snake_case for SQLAlchemy attributes
            attr = col.lower().replace(" ", "_")
            setattr(record, attr, val)
        mock_records.append(record)

    # Create an asymc mock session
    mock_session = AsyncMock()
    mock_result = mocker.MagicMock()
    mock_result.scalars.return_value.all.return_value = mock_records
    mock_session.execute.return_value = mock_result

    # Act
    df_result = await load_raw_data(mock_session, RawData)

    # Assert
    assert isinstance(df_result, pd.DataFrame)
    assert df_result.shape[0] == booking_data.shape[0]
    assert "booking_id" in df_result.columns
    assert "_sa_instance_state" not in df_result.columns


def test_preprocess_data(booking_data, sample_processor):
    # Arrange
    X = booking_data.drop(columns=[DEPENDENT_VAR_NAME])
    y = booking_data[DEPENDENT_VAR_NAME]

    # Action
    X_train, X_test, y_train, y_test = preprocess_data(X, y, sample_processor)

    # Assert that the processor's fit_transform method was called
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)

    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)


@pytest.mark.asyncio
async def test_save_to_silver_db(preprocessed_booking_data):
    # Arrange
    X, y = preprocessed_booking_data

    X_train = X.iloc[: int(len(X) * 0.7)].copy()
    X_test = X.iloc[int(len(X) * 0.7) :].copy()

    y_train = y.iloc[: int(len(y) * 0.7)].copy()
    y_test = y.iloc[int(len(y) * 0.7) :].copy()

    # Use MagicMock for non-awaited methods to avoid warnings
    mock_session = AsyncMock()
    mock_session.add_all = MagicMock()
    mock_session.commit = AsyncMock()

    # Act
    await save_to_silver_db(X_train, y_train, X_test, y_test, mock_session)

    # Assert
    # call_count = 2 b/c there is a call for SilverDB's TrainValidationData and TestData
    assert mock_session.add_all.call_count == 2
    mock_session.commit.assert_called_once()


def test_build_pipeline():
    pipe = build_pipeline()
    assert isinstance(pipe, NoVacancyPipeline)


# capsys is a builtin pytest fixture that captures output during the test
def test_evaluate_model(capsys):
    # Arrange
    mock_pipe = MagicMock()
    mock_pipe.predict_proba.return_value = np.array([[0.2, 0.8], [0.3, 0.7]])
    y_test = pd.Series([1, 0])

    # Act
    evaluate_model(mock_pipe, pd.DataFrame(), y_test)

    # capture printed output
    output = capsys.readouterr().out
    assert "AUC" in output
    assert mock_pipe.predict_proba.called


@pytest.mark.asyncio
async def test_train_pipeline_logs_to_mlflow(monkeypatch, booking_data):
    """
    Test that MLflow logging works and pipeline components integrate correctly.
    """
    # Use a temp directory for MLflow tracking URI
    temp_dir = tempfile.mkdtemp()
    mlruns_dir = Path(temp_dir) / "mlruns"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{mlruns_dir}")

    # Use raw data since train_pipeline preprocesses internally
    raw_data_mock = booking_data.copy()

    with (
        patch.object(trainer.bronze_db, "create_session") as mock_bronze_session,
        patch.object(trainer.silver_db, "create_session") as mock_silver_session,
        patch("services.trainer.load_raw_data") as mock_load_raw_data,
        patch("services.trainer.save_to_silver_db") as mock_save_to_silver_db,
    ):

        # Configure mocks to return our raw test data
        mock_load_raw_data.return_value = raw_data_mock
        mock_save_to_silver_db.return_value = None

        # Properly mock the double-call pattern bronze_db.create_session()()
        # The real code does: async with bronze_db.create_session()() as session:
        # This means create_session() returns a sessionmaker, then () calls it
        mock_session = AsyncMock()
        mock_sessionmaker = AsyncMock()
        mock_sessionmaker.return_value = mock_session

        # Setup session managers that return AsyncMock instances that support async context management
        mock_bronze_session.return_value = mock_sessionmaker
        mock_silver_session.return_value = mock_sessionmaker

        # Run the training pipeline
        await train_pipeline()

    # Verify MLflow logged everything correctly
    client = mlflow.tracking.MlflowClient()

    # Allow MLflow to store experiment data in the temp directory
    # Retry for up to 5 seconds to ensure the run is logged
    experiment = None
    for _ in range(10):
        try:
            experiment = client.get_experiment_by_name("NoVacancyModelTraining")
            if experiment:
                break
        except Exception:
            pass
        time.sleep(0.5)

    # Assert experiment exists
    assert experiment is not None, "Experiment was not registered in MLflow"
    experiment_id = experiment.experiment_id

    # Collect the records of the modeling session
    runs = client.search_runs(
        experiment_ids=[experiment_id], order_by=["start_time desc"]
    )
    assert runs, "No MLflow runs found"
    run = runs[0]

    # Assert artifacts exist
    run_id = run.info.run_id

    # Check model artifact
    model_path = mlflow.artifacts.download_artifacts(run_id, "model")
    assert Path(model_path, "MLmodel").exists(), "MLmodel artifact not found"

    # Check processor artifact
    processor_path = mlflow.artifacts.download_artifacts(
        run_id, "processor/processor.joblib"
    )
    assert Path(processor_path).exists(), "Processor artifact not found"
    processor = joblib.load(processor_path)
    assert hasattr(
        processor, "fit_transform"
    ), "Processor doesn't have expected methods"

    # Check input example artifact
    input_example_path = mlflow.artifacts.download_artifacts(
        run_id, "input_example/input_example.parquet"
    )
    assert Path(input_example_path).exists(), "Input example artifact not found"
    df = pd.read_parquet(input_example_path)
    assert not df.empty, "Input example is empty"

    # Assert metrics were logged
    logged_metrics = run.data.metrics
    assert "val_auc" in logged_metrics, "Validation AUC not logged"
    assert "test_auc" in logged_metrics, "Test AUC not logged"

    # Assert the AUC values are reasonable (between 0 and 1)
    assert 0 <= logged_metrics["val_auc"] <= 1, "Invalid validation AUC value"
    assert 0 <= logged_metrics["test_auc"] <= 1, "Invalid test AUC value"

    # Assert parameters were logged
    logged_params = run.data.params
    assert "model_version" in logged_params, "Model version not logged"
    assert "imputer_type" in logged_params, "Imputer type not logged"
    assert "encoder_type" in logged_params, "Encoder type not logged"

    # FIX: Use the correct variable names that match your patches
    mock_load_raw_data.assert_called_once()
    mock_save_to_silver_db.assert_called_once()

    # Verify database sessions were created
    mock_bronze_session.assert_called_once()
    mock_silver_session.assert_called_once()

    # Clean up temp mlruns directory
    shutil.rmtree(temp_dir)
