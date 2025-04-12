from unittest.mock import patch

import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from app.schemas.bronze import RawData
from app.services import DATA_PATHS
from app.services.trainer import load_raw_data, train_pipeline


def test_load_raw_data_from_bronze(mocker, booking_data):
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
    
    # Create a mock BronzeSessionLocal
    mock_session = mocker.MagicMock()
    mock_session.query.return_value.all.return_value = mock_records

    # Act
    df_result = load_raw_data(mock_session)

    # Assert
    assert isinstance(df_result, pd.DataFrame)
    assert df_result.shape[0] == booking_data.shape[0]
    assert "booking_id" in df_result.columns
    assert "_sa_instance_state" not in df_result.columns