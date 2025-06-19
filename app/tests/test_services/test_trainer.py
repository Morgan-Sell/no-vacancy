import shutil
import tempfile
import time
from unittest.mock import AsyncMock, MagicMock
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


@pytest.fixture.asyncio
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

    # Create a mock session
    mock_session = mocker.MagicMock()
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
async def test_save_to_silver_db(mocker):
    # Arrange
    X_mock = pd.DataFrame(
        [
            {
                "booking_id": f"id_{i}",
                "number_of_adults": 1,
                "number_of_children": 0,
                "number_of_weekend_nights": 1,
                "number_of_week_nights": 2,  # Fixed: removed 'days'
                "lead_time": 10,
                "type_of_meal": "Meal Plan 1",
                "car_parking_space": 0,
                "room_type": "Room_Type 1",
                "average_price": 100.0,
                "market_segment_type": "Online",
                "is_repeat_guest": 0,
                "num_previous_cancellations": 0,
                "num_previous_bookings_not_canceled": 0,
                "special_requests": 0,
                "month_of_reservation": "jan",
                "day_of_week": "Monday",
            }
            for i in range(2)
        ]
    )

    y_mock = pd.Series([1, 0])
    mock_session = AsyncMock()

    # Act
    await save_to_silver_db(X_mock.copy(), y_mock, X_mock.copy(), y_mock, mock_session)

    # Assert
    assert mock_session.add_all.call_count == 2
    assert mock_session.commit.called


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
async def test_train_pipeline_logs_to_mlflow(monkeypatch):
    # Use a temp directory for MLflow tracking URI
    temp_dir = tempfile.mkdtemp()
    mlruns_dir = Path(temp_dir) / "mlruns"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{mlruns_dir}")

    # Run the training pipeline
    await train_pipeline()

    client = mlflow.tracking.MlflowClient()

    # Load the latest run
    # Allow MLflow to store experiment data in the temp directory
    # Retry for up to 5 seconds to ensure the run is logged
    for _ in range(10):
        experiment = client.get_experiment_by_name("NoVacancyModelTraining")
        if experiment:
            break
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
    model_path = mlflow.artifacts.download_artifacts(run_id, "model")
    assert Path(model_path, "MLmodel").exists(), "MLmodel artifact not found"

    processor_path = mlflow.artifacts.download_artifacts(
        run_id, "processor/processor.joblib"
    )
    assert Path(processor_path).exists()
    processor = joblib.load(processor_path)
    assert hasattr(processor, "prepare_data")  # rough sanity check

    input_example_path = mlflow.artifacts.download_artifacts(
        run_id, "input_example/input_example.parquet"
    )
    assert Path(input_example_path).exists()
    df = pd.read_parquet(input_example_path)
    assert not df.empty

    # Assert metrics were logged
    logged_metrics = run.data.metrics
    assert "val_auc" in logged_metrics
    assert "test_auc" in logged_metrics

    # Clean up temp mlruns directory
    shutil.rmtree(temp_dir)
