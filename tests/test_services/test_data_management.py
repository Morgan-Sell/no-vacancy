import os
from pathlib import Path
from unittest.mock import Mock, patch

import joblib
import pytest

from app.services.data_management import DataManagement
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
        with patch("joblib.load", return_value={"pipeline": mock_pipeline, "processor": mock_processor}):
            loaded_pipeline, loaded_processor = dm.load_pipeline()
            assert isinstance(loaded_pipeline, NoVacancyPipeline)
            assert isinstance(loaded_processor, NoVacancyDataProcessing)
            
            # Check processor metadata
            assert hasattr(loaded_processor, "vars_to_drop"), "❌ Processor missing 'vars_to_drop'."
            assert hasattr(loaded_processor, "variable_rename"), "❌ Processor missing 'variable_rename'."
            assert hasattr(loaded_processor, "month_abbreviation"), "❌ Processor missing 'month_abbreviation'."
            
            # Functional validation
            try:
                # Apply transformation with the loaded processor
                X_test, _ = loaded_processor.transform(
                    booking_data.drop(columns=["booking status"]), 
                    booking_data["booking status"]
                )
                
                assert not X_test.empty, "❌ Transformed test data is empty."
                
                # Run prediction with the loaded pipeline
                predictions = loaded_pipeline.predict(X_test)
                assert predictions is not None, "❌ Pipeline failed to generate predictions."
            except Exception as e:
                pytest.fail(f"❌ Functional validation failed after loading artifacts: {e}")


def test_load_pipeline_not_found(dm):
    with patch.object(Path, "exists", return_value=False):
        with pytest.raises(
            FileNotFoundError,
            match="❌ Error during pipeline loading: Pipeline file not found at",
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
