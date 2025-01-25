import os
from pathlib import Path
from unittest.mock import Mock, patch

import joblib
import numpy as np
import pytest
from feature_engine.encoding import OneHotEncoder
from feature_engine.imputation import CategoricalImputer
from sklearn.ensemble import RandomForestClassifier

from app.services import VARS_TO_IMPUTE, VARS_TO_OHE
from app.services.pipeline import NoVacancyPipeline
from app.services.preprocessing import NoVacancyDataProcessing


# -----------------------------
# Test: save_pipeline
# -----------------------------
def test_save_pipeline_success(dm, mock_pipeline, mock_processor):
    # Arrange
    with patch("joblib.dump") as mock_dump:
        # Act
        dm.save_pipeline(mock_pipeline, mock_processor)

        # Assert
        mock_dump.assert_called_once_with(
            {"pipeline": mock_pipeline, "processor": mock_processor}, dm.pipeline_path
        )


def test_save_pipeline_invalid_pipeline(dm, mock_processor):
    with pytest.raises(
        TypeError,
        match="❌ Error during pipeline saving: The pipeline to be saved must be an instance of NoVacancyPipeline",
    ):
        dm.save_pipeline("Wrong pipe!", mock_processor)


def test_save_pipeline_invalid_processor(dm, mock_pipeline):
    with pytest.raises(
        TypeError,
        match="❌ Error during pipeline saving: The processor to be saved must be an instance of NoVacancyDataProcessing",
    ):
        dm.save_pipeline(mock_pipeline, "Wrong processor!")


def test_save_pipeline_exception(dm, mock_pipeline, mock_processor):
    with patch("joblib.dump", side_effect=Exception("BOOM!")):
        with pytest.raises(Exception, match="❌ Error during pipeline saving: BOOM!"):
            dm.save_pipeline(mock_pipeline, mock_processor)


# -----------------------------
# Test: load_pipeline
# -----------------------------
def test_load_pipeline_success(dm, mock_pipeline, mock_processor, booking_data):
    # patch.object() mocks an attribute or method of a specific class, rather than a global reference in a module
    with patch.object(Path, "exists", return_value=True):
        with patch(
            "joblib.load",
            return_value={"pipeline": mock_pipeline, "processor": mock_processor},
        ):
            loaded_pipeline, loaded_processor = dm.load_pipeline()

            # Check data types
            assert isinstance(loaded_pipeline, NoVacancyPipeline)
            assert isinstance(loaded_processor, NoVacancyDataProcessing)

            # Check methods
            assert hasattr(loaded_pipeline, "predict")
            assert hasattr(loaded_processor, "transform")

            # Check processor metadata
            assert hasattr(
                loaded_processor, "vars_to_drop"
            ), "❌ Processor missing 'vars_to_drop'."
            assert hasattr(
                loaded_processor, "variable_rename"
            ), "❌ Processor missing 'variable_rename'."
            assert hasattr(
                loaded_processor, "month_abbreviation"
            ), "❌ Processor missing 'month_abbreviation'."


def test_load_pipeline_not_found(dm):
    with patch.object(Path, "exists", return_value=False):
        with pytest.raises(
            FileNotFoundError,
            match="❌ Error during pipeline loading: Pipeline file not found at",
        ):
            dm.load_pipeline()


def test_load_pipeline_invalid_artifacts(dm):
    with patch.object(Path, "exists", return_value=True):
        with patch(
            "joblib.load", return_value={"pipeline": "Invalid", "processor": "Invalid"}
        ):
            with pytest.raises(
                TypeError,
                match="❌ Error during pipeline loading: Loaded pipeline is not an instance of NoVacancyPipeline",
            ):
                dm.load_pipeline()


def test_load_pipeline_exception(dm):
    # Mock dm.pipeline_path.exists() so FileNotFoundError is not raised.
    with patch.object(Path, "exists", return_value=True):
        with patch("joblib.load", side_effect=Exception("Explode!")):
            with pytest.raises(
                Exception, match="❌ Error during pipeline loading: Explode!"
            ):
                dm.load_pipeline()


# -----------------------------
# Test: delete_pipeline
# -----------------------------
def test_delete_pipeline_success(dm, temp_pipeline_path):
    # Simulate an existing pipeline file
    temp_pipeline_path.touch()
    assert temp_pipeline_path.exists() is True

    dm.delete_pipeline()
    assert temp_pipeline_path.exists() is False


def test_delete_pipeline_not_found(dm):
    with pytest.raises(
        FileNotFoundError,
        match="❌ Error during pipeline deletion: Pipeline file not found at",
    ):
        dm.delete_pipeline()


def test_delete_pipeline_exception(dm, temp_pipeline_path):
    # Simulate an existing pipeline file
    temp_pipeline_path.touch()

    with patch.object(Path, "unlink", side_effect=Exception("OH NO!")):
        with pytest.raises(
            Exception, match="❌ Error during pipeline deletion: OH NO!"
        ):
            dm.delete_pipeline()


# ---------------------------------
# Validate Pipeline and Processor
# ---------------------------------
def test_validate_pipeline_and_processor_valid(dm, sample_pipeline, sample_processor):
    """Test successful validation of pipeline and processor."""
    dm._validate_pipeline_and_processor(sample_pipeline, sample_processor)


def test_validate_pipeline_and_processor_invalid_pipeline(dm, sample_processor):
    """Test validation with an invalid pipeline."""
    with pytest.raises(
        TypeError,
        match="❌ Error during pipeline validation: The pipeline must be an instance of NoVacancyPipeline",
    ):
        dm._validate_pipeline_and_processor("Invalid Pipeline", sample_processor)


def test_validate_pipeline_and_processor_invalid_processor(dm, sample_pipeline):
    """Test validation with an invalid processor."""
    with pytest.raises(
        TypeError,
        match="❌ Error during pipeline validation: The processor must be an instance of NoVacancyDataProcessing",
    ):
        dm._validate_pipeline_and_processor(sample_pipeline, "Invalid Processor")


# -----------------------------------------
# Integration b/t Pipeline and Processor
# -----------------------------------------
def test_pipeline_processor_integration(booking_data, sample_processor):
    # Transform data
    X = booking_data.drop(columns=["booking status"])
    y = booking_data["booking status"]
    X_prcsd, y_prcsd = sample_processor.transform(X, y)

    # Define the pipeline attributes
    imputer = CategoricalImputer(imputation_method="frequent", variables=VARS_TO_IMPUTE)
    encoder = OneHotEncoder(variables=VARS_TO_OHE)
    clsfr = RandomForestClassifier()

    # Define the search space for RCSV
    search_space = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [3, 5, 7],
    }

    # Initialize the pipeline
    pipeline = NoVacancyPipeline(imputer, encoder, clsfr)
    pipeline.pipeline(search_space)

    # Fit pipeline
    pipeline.fit(X_prcsd, y_prcsd)

    # Predict
    predictions = pipeline.predict(X_prcsd)
    probs = pipeline.predict_proba(X_prcsd)

    # Assert
    assert len(predictions) == len(X_prcsd)
    assert isinstance(predictions, np.ndarray)

    assert probs.shape == (len(X_prcsd), 2)
    assert np.all((probs >= 0) & (probs <= 1))

    assert pipeline.rscv.best_estimator_ is not None
