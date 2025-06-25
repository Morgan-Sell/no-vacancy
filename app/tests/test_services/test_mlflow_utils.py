import pytest
from unittest.mock import patch, MagicMock
from services.mlflow_utils import MLflowArtifactLoader
from services import MLFLOW_EXPERIMENT_NAME, MLFLOW_PROCESSOR_PATH
import mlflow
import joblib


class TestMLflowArtifactLoader:
    @pytest.fixture
    def loader(self, mock_mlflow_client, mocker):
        """Create MLFlowArtifactLoader instance with mocked MLflow."""

        mocker.patch.object(mlflow, "set_tracking_uri")
        mocker.patch.object(mlflow, "MlflowClient")

        loader = MLflowArtifactLoader()
        loader.client = mock_mlflow_client.return_value
        return loader

    def test_load_pipeline_artifacts_by_stage_sucesss(
        self, loader, mock_mlflow_pipeline, mock_mlflow_processor, mocker
    ):
        # Arrange
        mock_version = MagicMock()
        mock_version.run_id = "test_run_id"
        mock_version.version = "23"
        loader.client.get_latest_versions.return_value = [mock_version]

        mock_load_model = mocker.patch.object(mlflow.sklearn, "load_model")
        mock_download = mocker.patch.object(mlflow.artifacts, "download_artifacts")
        mock_joblib_load = mocker.patch.object(joblib, "load")

        mock_load_model.return_value = mock_mlflow_pipeline
        mock_download.return_value = "/tmp/no_vacancy_processor.pkl"
        mock_joblib_load.return_value = mock_mlflow_processor

        # Act
        pipeline, processor, model_version = loader.load_pipeline_artifacts_by_stage(
            "Production"
        )

        # Assert
        loader.client.get_latest_versions.assert_called_once_with(
            MLFLOW_EXPERIMENT_NAME, stages=["Production"]
        )
        mock_load_model.assert_called_once_with(
            model_uri=f"models://{MLFLOW_EXPERIMENT_NAME}/Production"
        )
        mock_download.assert_called_once_with(
            run_id="test_run_id", artifact_path=MLFLOW_PROCESSOR_PATH
        )
        mock_joblib_load.assert_called_once_with("/tmp/no_vacancy_processor.pkl")

        # Assert
        assert pipeline == mock_mlflow_pipeline
        assert processor == mock_mlflow_processor
        assert model_version == "23"

    def test_load_pipeline_artifacts_by_version_success(
        self, loader, mock_mlflow_pipeline, mock_mlflow_processor, mocker
    ):
        # Arrange
        mock_version_obj = MagicMock()
        mock_version_obj.run_id = "forest-gump"
        loader.client.get_model_version.return_value = mock_version_obj

        mock_load_model = mocker.patch.object(mlflow.sklearn, "load_model")
        mock_download = mocker.patch.object(mlflow.artifacts, "download_artifacts")
        mock_joblib_load = mocker.patch.object(joblib, "loadl")

        mock_load_model.return_value = mock_mlflow_pipeline
        mock_download.return_value = "/tmp/no_vacancy_processor.pkl"
        mock_joblib_load.return_value = mock_mlflow_processor

        # Act
        pipeline, processor = loader.load_pipeline_artifacts_by_version("36")

        # Assert
        loader.client.get_model_version.assert_called_once_with(
            name=MLFLOW_EXPERIMENT_NAME, version="36"
        )
        mock_load_model.assert_called_once_with(
            model_uri=f"models://{MLFLOW_EXPERIMENT_NAME}/36"
        )
        mock_download.assert_called_once_with(
            run_id="forest-gump", artifact_path=MLFLOW_PROCESSOR_PATH
        )

        assert pipeline == mock_mlflow_pipeline
        assert processor == mock_mlflow_processor

    def test_get_artifact_metadata_success(self, loader):
        # Arrange
        mock_version = MagicMock()
        mock_version.version = "84"
        mock_version.run_id = "hemmingway"
        mock_version.current_stage = "Production"
        mock_version.creation_timestamp = 1234567890

        mock_run = MagicMock()
        mock_run.data.metrics = {"accuracy": 0.93, "auc": 0.88}
        mock_run.info.artifact_uri = "s3://bucket/path/to/artifacts"

        loader.client.get_latest_versions.return_value = [mock_version]
        loader.client.get_run.return_value = mock_run

        # Act
        metadata = loader.get_artifact_metadata("Production")

        # Assert
        expected_metadata = {
            "version": "84",
            "run_id": "hemmingway",
            "stage": "Production",
            "metrics": {"accuracy": 0.93, "auc": 0.88},
            "artifacts": "s3://bucket/path/to/artifacts",
            "created_at": 1234567890,
        }
        assert metadata == expected_metadata

    @pytest.mark.parametrize(
        "tags,stage,expected",
        [
            ({"manual_validation": "approved"}, "Staging", True),
            ({"data_scientist_approved": "true"}, "Staging", True),
            ({"validated": "true"}, "Staging", True),
            ({"ready_for_production": "true"}, "Staging", True),
            ({"manual_validation": "approved"}, "Production", False),  # Wrong stage
            ({"manual_validation": "rejected"}, "Staging", False),  # Not approved
            ({}, "Staging", False),  # No validation tags
        ],
    )
    def test_check_manual_validation_status(self, loader, tags, stage, expected):
        # Arrange
        mock_model_details = MagicMock()
        mock_model_details.tags = tags
        mock_model_details.current_stage = stage
        loader.client.get_model_version.return_value = mock_model_details

        # Act
        result = loader.check_manual_validation_status("42")

        # Assert
        assert result == expected

    def test_promote_to_production(self, loader):
        # Act
        loader.promote_to_production("777")

        # Assert
        loader.client.transition_model_version_stage.assert_called_once_with(
            name=MLFLOW_EXPERIMENT_NAME,
            version="777",
            stage="Production",
            archive_existing_versions=True,
        )

    def test_get_latest_production_model(self, loader):
        # Arrange
        with patch.object(loader, "get_artifact_metadata") as mock_get_metadata:
            mock_get_metadata.return_value = {"version": "13"}

            # Act
            version = loader.get_latest_production_model()

            # Assert
            mock_get_metadata.assert_called_once_with(stage="Production")
            assert version == "13"
