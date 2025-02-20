from unittest.mock import patch

import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from app.services import DATA_PATHS
from app.services.trainer import train_pipeline


# TODO: Need to update once data source changes, i.e., csv -> Postgres
def test_train_pipeline_end_to_end(temp_booking_data_csv, mock_logger):
    # Arrange
    # Point DATA_PATHS to booking_data fixture
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "app.services.trainer.DATA_PATHS", {"raw_data": temp_booking_data_csv}
        )

        # Act
        train_pipeline()

    # Assert
    # Ensure the logger correctly recorded the AUC score
    mock_logger.info.assert_called_once()
    logged_message = mock_logger.info.call_args[0][0]
    assert "AUC" in logged_message, "AUC score was incorrectly logged."
