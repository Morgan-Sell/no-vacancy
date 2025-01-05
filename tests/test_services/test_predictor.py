from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.services.predictor import make_prediction


def test_make_prediction_success(booking_data):
    # Arrange: Mock NoVacancyPipeline return values
    mock_pipeline = MagicMock()
    mock_pipeline.predict.return_value = [0, 1, 0]
    mock_pipeline.predict_proba.return_value = [[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]]

    with patch(
        "app.services.data_management.DataManagement.load_pipeline",
        return_value=mock_pipeline,
    ):
        # Act
        results = make_prediction(booking_data)

        # Assert
        assert isinstance(results, pd.DataFrame)
        assert results.shape[0] == 3
        assert results.columns.tolist() == [
            "prediction",
            "probability_not_canceled",
            "probabilities_canceled",
        ]


def test_make_prediction_with_empty_data():
    # Arrange
    empty_data = pd.DataFrame()

    with pytest.raises(ValueError, match="❌ Invalid input: Input data is empty. Cannot make predictions on an empty DataFrame."):
        make_prediction(empty_data)