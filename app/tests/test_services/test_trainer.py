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
