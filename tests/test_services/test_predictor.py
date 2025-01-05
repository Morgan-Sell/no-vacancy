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

    # Act & Assert
    with pytest.raises(
        ValueError,
        match="❌ Invalid input: Input data is empty. Cannot make predictions on an empty DataFrame.",
    ):
        make_prediction(empty_data)


def test_make_prediction_pipeline_not_found(booking_data):
    # Arrange
    with patch(
        "app.services.data_management.DataManagement.load_pipeline",
        side_effect=FileNotFoundError("Pipeline not found"),
    ):
        # Act & Assert
        with pytest.raises(
            FileNotFoundError, match="❌ No pipeline found: Pipeline not found"
        ):
            make_prediction(booking_data)


def test_make_prediction_unexpected_error(booking_data):
    # Arrange
    mock_pipeline = MagicMock()
    mock_pipeline.predict.side_effect = Exception("Unexpected prediction error")
    mock_pipeline.predict_proba.return_vale = [[0.8, 0.2], [0.3, 0.7]]

    with patch(
        "app.services.data_management.DataManagement.load_pipeline",
        return_value=mock_pipeline,
    ):
        # Act & Assert
        with pytest.raises(
            RuntimeError, match="❌ Prediction failed: Unexpected prediction error"
        ):
            make_prediction(booking_data)


def test_make_prediction_invalid_input_type():
    # Arrange
    invalid_input = {"breakfast": [3, "waffles"]}

    with pytest.raises(
        ValueError, match="❌ Invalid input: Input must be a pandas DataFrame"
    ):
        make_prediction(invalid_input)


def test_make_prediction_single_observation(booking_data):
    # Arrange
    single_observation = booking_data.iloc[0].to_frame().T.copy()

    mock_pipeline = MagicMock()
    mock_pipeline.predict.return_value = [1]
    mock_pipeline.predict_proba.return_value = [[0.3, 0.7]]

    with patch(
        "app.services.data_management.DataManagement.load_pipeline",
        return_value=mock_pipeline,
    ):
        # Act
        results = make_prediction(single_observation)

        # Assert
        assert isinstance(results, pd.DataFrame)
        assert results.shape[0] == 1
        assert results.columns.tolist() == [
            "prediction",
            "probability_not_canceled",
            "probabilities_canceled",
        ]
