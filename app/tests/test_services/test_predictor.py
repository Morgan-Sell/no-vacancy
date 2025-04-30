from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from services.predictor import make_prediction


def test_make_prediction_success(booking_data, mock_pipeline, mock_processor, pm):
    # Arrange: Create mock rscv and best_estimator_
    mock_rscv = MagicMock()
    mock_best_estimator = MagicMock()

    # Mock the named_steps of the best estimator
    mock_best_estimator.named_steps = {
        "imputation_step": MagicMock(),
        "encoding_step": MagicMock(
            get_feature_names_out=MagicMock(return_value=["col1", "col2"])
        ),
    }

    # Assign best_estimator_ to mock_rscv
    mock_rscv.best_estimator_ = mock_best_estimator

    # Attach rscv to mock_pipeline
    mock_pipeline.rscv = mock_rscv

    with patch(
        "app.services.pipeline_management.PipelineManagement.load_pipeline",
        return_value=(mock_pipeline, mock_processor),
    ):
        # Act
        results = make_prediction(booking_data, pm)

        # Assert
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 2, "Expected 2 predictions from mock_pipeline."
        assert list(results.columns) == [
            "prediction",
            "probability_not_canceled",
            "probabilities_canceled",
        ], "Unexpected columns in prediction results."


def test_make_prediction_with_empty_data():
    # Arrange
    empty_data = pd.DataFrame()

    # Act & Assert
    with pytest.raises(
        ValueError,
        match="❌ Invalid input: Input data is empty. Cannot make predictions on an empty DataFrame.",
    ):
        make_prediction(empty_data)


def test_make_prediction_pipeline_not_found(booking_data, mock_processor, pm):
    # Arrange: Patch PipelineManagement to raise FileNotFoundError
    with patch(
        "app.services.pipeline_management.PipelineManagement.load_pipeline",
        side_effect=FileNotFoundError("Pipeline not found"),
    ):
        # Act & Assert
        with pytest.raises(
            FileNotFoundError, match="❌ No pipeline found: Pipeline not found"
        ):
            make_prediction(booking_data, pm)


def test_make_prediction_unexpected_error(
    booking_data, mock_pipeline, mock_processor, pm
):
    # Arrange: Create a mock RandomizedSearchCV with best_estimator_
    mock_rscv = MagicMock()
    mock_best_estimator = MagicMock()

    # Mock named_steps in best_estimator_ to avoid AttributeError
    mock_best_estimator.named_steps = {
        "imputation_step": MagicMock(),
        "encoding_step": MagicMock(
            get_feature_names_out=MagicMock(return_value=["col1", "col2"])
        ),
    }

    # Attach the best_estimator_ to mock_rscv
    mock_rscv.best_estimator_ = mock_best_estimator

    # Simulate the unexpected error during prediction
    mock_pipeline.rscv = mock_rscv
    mock_pipeline.predict.side_effect = Exception("Unexpected prediction error")

    # Patch PipelineManagement to return the mock pipeline and processor
    with patch(
        "app.services.pipeline_management.PipelineManagement.load_pipeline",
        return_value=(mock_pipeline, mock_processor),
    ):
        # Act & Assert
        with pytest.raises(
            RuntimeError, match="❌ Prediction failed: Unexpected prediction error"
        ):
            make_prediction(booking_data, pm)


def test_make_prediction_invalid_input_type():
    # Arrange
    invalid_input = {"breakfast": [3, "waffles"]}

    with pytest.raises(
        ValueError, match="❌ Invalid input: Input must be a pandas DataFrame"
    ):
        make_prediction(invalid_input)


def test_make_prediction_single_observation(
    booking_data, mock_pipeline, mock_processor, pm
):
    # Use a single observation
    single_observation = booking_data.iloc[0].to_frame().T.copy()

    # Mock rscv and best_estimator_
    mock_rscv = MagicMock()
    mock_best_estimator = MagicMock()

    # Simulate named_steps for best_estimator_
    mock_best_estimator.named_steps = {
        "imputation_step": MagicMock(),
        "encoding_step": MagicMock(
            get_feature_names_out=MagicMock(return_value=["col1", "col2"])
        ),
    }

    # Attach mocks
    mock_rscv.best_estimator_ = mock_best_estimator
    mock_pipeline.rscv = mock_rscv

    # Mock prediction behavior
    mock_pipeline.predict.return_value = [1]
    mock_pipeline.predict_proba.return_value = [[0.3, 0.7]]

    with patch(
        "app.services.pipeline_management.PipelineManagement.load_pipeline",
        return_value=(mock_pipeline, mock_processor),
    ):
        # Act
        results = make_prediction(single_observation, pm)

        # Assert
        assert isinstance(results, pd.DataFrame)
        assert results.shape[0] == 1
        assert results.columns.tolist() == [
            "prediction",
            "probability_not_canceled",
            "probabilities_canceled",
        ]
