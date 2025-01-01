import os
from pathlib import Path
from unittest.mock import Mock, patch

import joblib
import pytest

from app.services.data_management import DataManagement


# -----------------------------
# Test: save_pipeline
# -----------------------------
def test_save_pipeline_success(dm, sample_pipeline, temp_pipeline_path):
    # Arrange: adjust temp_pipeline_path to match PIPELINE_SAVE_FILE in config.py
    expected_path = temp_pipeline_path.parent / "no_vacancy_pipeline"

    
    with patch("joblib.dump") as mock_joblib_dump:
        # Act
        dm.save_pipeline(sample_pipeline)
        
        # Arrange
        mock_joblib_dump.assert_called_once_with(sample_pipeline, expected_path)


def test_save_pipeline_failure(dm, sample_pipeline, temp_pipeline_path):

    with patch("joblib.dump") as mock_joblib_dump:
        mock_joblib_dump.side_effect = Exception("Boom!")
        
        # Act & Assert
        with pytest.raises(RuntimeError, match="❌ Failed to save pipeline: Boom!"):
            dm.save_pipeline(sample_pipeline)


# -----------------------------
# Test: load_pipeline
# -----------------------------
def test_load_pipeline_success(dm, sample_pipeline, temp_pipeline_path):
    # Arrange
    expected_path = temp_pipeline_path.parent / "no_vacancy_pipeline"
    
    with patch("joblib.load", return_value=sample_pipeline) as mock_joblib_load:
        joblib.dump(sample_pipeline, expected_path)

        # Act
        result = dm.load_pipeline()

        # Assert
        mock_joblib_load.assert_called_once_with(expected_path)
        assert result == sample_pipeline


def test_load_pipeline_not_found(dm, temp_pipeline_path):
    # Arrange
    expected_path = temp_pipeline_path.parent / "no_vacancy_pipeline"

    if expected_path.exists():
        os.remove(expected_path)
    
    # Act & Assert
    with pytest.raises(FileNotFoundError, match=f"❌ No pipeline found at {expected_path}"):
        dm.load_pipeline()


def test_load_pipeline_failure(dm):
    # Arrange: Ensure self.pipeline_path exists otherwise FileNotFoundError will be raised
    with patch("pathlib.Path.exists", return_value=True):
        with patch("joblib.load", side_effect=Exception("Explode!")):

            # Act & Assert
            with pytest.raises(RuntimeError, match="❌ Failed to load pipeline: Explode!"):
                dm.load_pipeline()

# -----------------------------
# Test: delete_pipeline
# -----------------------------
def test_delete_pipeline_success(dm, temp_pipeline_path):
    # Arrange
    expected_path = temp_pipeline_path.parent / "no_vacancy_pipeline"
    expected_path.touch()  # Create an empty file to simulate an existing pipeline

    with patch("os.remove") as mock_remove:
        # Act
        dm.delete_pipeline()

        # Assert
        mock_remove.assert_called_once_with(expected_path)


def test_delete_pipeline_not_found(dm, temp_pipeline_path):
    # Arrange
    expected_path = temp_pipeline_path.parent / "no_vacancy_pipeline"

    with patch.object(dm.logger, "warning") as mock_warning:
        # Act
        dm.delete_pipeline()

        # Assert
        mock_warning.assert_called_once_with(
            f"❌ No pipeline found to delete at {expected_path}"
        )

def test_delete_pipeline_failure(dm, temp_pipeline_path):
    # Arrange
    expected_path = temp_pipeline_path.parent / "no_vacancy_pipeline"
    expected_path.touch() # create an empty file to simulate an existig pipeline

    with patch("os.remove", side_effect=Exception("Kaboom!")):
        with pytest.raises(RuntimeError, match="❌ Failed to delete pipeline: Kaboom!"):
            # Act & Assert
            dm.delete_pipeline()

