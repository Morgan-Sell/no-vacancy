from unittest.mock import patch

import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from app.services.config_services import DATA_PATHS
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


# def test_train_pipeline_success(
#     temp_booking_data_csv, mock_pipeline, mock_logger, mocker
# ):
#     """
#     Test that train_pipeline executes successfully:
#     - Data is loaded from a temporary CSV file
#     - Pipeline is trained
#     - Predictions are made
#     """
#     # Arrange: Mock the DATA_PATHS to point to the temporary file
#     mocker.patch("app.services.trainer.DATA_PATHS", {"raw_data": temp_booking_data_csv})
#     mocker.patch("pandas.read_csv", return_value=pd.read_csv(temp_booking_data_csv))

#     # Action
#     train_pipeline()

#     # Assert
#     mock_pipeline.assert_called_once()  # NoVacancyPipeline is instantiated exactly once
#     mock_pipeline.fit.assert_called_once()  # fit method is called exactly once
#     mock_pipeline.predict_proba.assert_called_once()  # predict_proba method is called exactly once
#     mock_logger.assert_called()  # logger.info is called at least once
