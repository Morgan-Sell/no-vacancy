import pandas as pd
from sklearn.metrics import roc_auc_score
from app.services.trainer import train_pipeline
from app.services.config_services import DATA_PATHS


def test_train_pipeline_success(
    temp_booking_data_csv, mock_pipeline, mock_logger, mocker
):
    """
    Test that train_pipeline executes successfully:
    - Data is loaded from a temporary CSV file
    - Pipeline is trained
    - Predictions are made
    """
    # Arrange: Mock the DATA_PATHS to point to the temporary file
    mocker.patch("app.services.trainer.DATA_PATHS", {"raw_data": temp_booking_data_csv})
    mocker.patch("pandas.read_csv", return_value=pd.read_csv(temp_booking_data_csv))

    # Action
    train_pipeline()

    # Assert
    mock_pipeline.assert_called_once()  # NoVacancyPipeline is instantiated exactly once
    mock_pipeline.fit.assert_called_once()  # fit method is called exactly once
    mock_pipeline.predict_proba.assert_called_once()  # predict_proba method is called exactly once
    mock_logger.assert_called()  # logger.info is called at least once
