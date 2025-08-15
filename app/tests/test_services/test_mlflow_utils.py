import pytest
from unittest.mock import patch, MagicMock
from services.mlflow_utils import MLflowArtifactLoader
from services import MLFLOW_EXPERIMENT_NAME, MLFLOW_PROCESSOR_PATH
import mlflow
import joblib
import os


class TestMLflowArtifactLoader:
    """Essential tests for MLflowArtifactLoader."""

    def test_init_sets_tracking_uri_and_client(self, mock_mlflow_artifact_setup):
        """Test that initialization properly sets tracking URI and creates client."""
        mock_set_uri, mock_client_class = mock_mlflow_artifact_setup

        with patch.dict(os.environ, {"MLFLOW_TRACKING_URI": "http://test:5000"}):
            loader = MLflowArtifactLoader()

            mock_set_uri.assert_called_once_with("http://test:5000")
            mock_client_class.assert_called_once()
            assert loader.client == mock_client_class.return_value

    def test_load_pipeline_artifacts_by_alias_success(
        self, mlflow_loader, mock_model_version
    ):
        """Test successful loading of pipeline artifacts by alias."""
        # Arrange
        mock_client = mlflow_loader.client
        mock_client.get_model_version_by_alias.return_value = mock_model_version

        mock_pipeline = MagicMock()
        mock_processor = MagicMock()

        with (
            patch(
                "services.mlflow_utils.mlflow.sklearn.load_model",
                return_value=mock_pipeline,
            ),
            patch(
                "services.mlflow_utils.mlflow.artifacts.download_artifacts",
                return_value="/tmp/fake_processor.pkl",
            ),
            patch("joblib.load", return_value=mock_processor),
        ):

            # Act
            pipeline, processor, version = (
                mlflow_loader.load_pipeline_artifacts_by_alias("production")
            )

            # Assert
            assert pipeline == mock_pipeline
            assert processor == mock_processor
            assert version == "7"

    def test_load_pipeline_artifacts_by_alias_no_model_found(self, mlflow_loader):
        """Test error handling when no model is found."""
        # Arrange
        mlflow_loader.client.get_model_version_by_alias.return_value = None

        # Act & Assert
        with pytest.raises(
            RuntimeError, match="No model version found with alias 'staging'"
        ):
            mlflow_loader.load_pipeline_artifacts_by_alias("staging")

    def test_promote_to_production_success(
        self, mlflow_loader, mock_model_version, capsys
    ):
        """Test successful promotion to production."""
        # Arrange
        mlflow_loader.client.get_model_version_by_alias.return_value = (
            mock_model_version
        )

        # Act
        mlflow_loader.promote_to_production("9")

        # Assert
        mlflow_loader.client.set_registered_model_alias.assert_called_with(
            name=MLFLOW_EXPERIMENT_NAME, alias="production", version="9"
        )

        captured = capsys.readouterr()
        assert "✅ Model version 9 promoted to production" in captured.out

    def test_promote_to_production_no_existing_production(self, mlflow_loader, capsys):
        """Test promotion when no existing production model exists."""
        # Arrange
        mlflow_loader.client.get_model_version_by_alias.return_value = None

        # Act
        mlflow_loader.promote_to_production("11")

        # Assert
        mlflow_loader.client.delete_registered_model_alias.assert_not_called()
        mlflow_loader.client.set_registered_model_alias.assert_called_with(
            name=MLFLOW_EXPERIMENT_NAME, alias="production", version="11"
        )

    def test_check_manual_validation_status_approved(self, mlflow_loader):
        """Test validation status check for approved model."""
        # Arrange
        mock_version = MagicMock()
        mock_version.tags = {"validation_status": "approved"}
        mock_version.version = "5"
        mock_version.aliases = ["staging"]

        mlflow_loader.client.get_model_version_by_alias.return_value = mock_version
        mlflow_loader.client.get_model_version.return_value = mock_version

        # Act
        is_validated = mlflow_loader.check_manual_validation_status_by_alias("staging")

        # Assert
        assert is_validated is True

    def test_check_manual_validation_status_not_approved(self, mlflow_loader):
        """Test validation status check non-approved model."""
        # Arrange
        mock_version = MagicMock()
        mock_version.tags = {"validation_status", "pending"}
        mock_version.version = "3"
        mock_version.aliases = ["staging"]

        mlflow_loader.client.get_model_version_by_alias.return_value = mock_version
        mlflow_loader.client.get_model_version.return_value = mock_version

        # Act
        is_validated = mlflow_loader.check_manual_validation_status_by_alias("staging")

        # Assert
        assert is_validated is False

    @pytest.mark.parametrize(
        "tag_key,tag_value",
        [
            ("validation_status", "approved"),
            ("model_approval", "approved"),
            ("deployment_approval", "approved"),
            ("ModelApprovalStatus", "Approved"),
        ],
    )
    def test_check_manual_validation_various_tags(
        self, mlflow_loader, tag_key, tag_value
    ):
        """Test validation status with various approval tag formats."""
        # Arrange
        mock_version = MagicMock()
        mock_version.tags = {tag_key: tag_value}
        mock_version.version = "5"
        mock_version.aliases = ["staging"]

        mlflow_loader.client.get_model_version_by_alias.return_value = mock_version
        mlflow_loader.client.get_model_version.return_value = mock_version

        # Act
        is_validated = mlflow_loader.check_manual_validation_status_by_alias("staging")

        # Assert
        assert is_validated is True

    def test_get_artifact_metadata_by_alias_success(
        self, mlflow_loader, mock_model_version
    ):
        """Test successful retrieval of artifact metadata"""
        # Arrange
        mlflow_loader.client.get_model_version_by_alias.return_value = (
            mock_model_version
        )

        mock_run = MagicMock()
        mock_run.data.metrics = {"auc": 0.95, "accuracy": 0.89}
        mock_run.info.artifact_uri = "s3://bucket/artifacts"
        mlflow_loader.client.get_run.return_value = mock_run

        # Act
        metadata = mlflow_loader.get_artifact_metadata_by_alias("production")

        # Assert
        assert (
            metadata["version"] == "7"
        )  # from mock_model_version fixture in conftest.py
        assert metadata["alias"] == "production"
        assert metadata["metrics"]["auc"] == 0.95
        assert metadata["artifacts"] == "s3://bucket/artifacts"

    def test_list_all_aliases_success(self, mlflow_loader):
        """Test successful listing of all aliases."""
        # Arrange
        mock_version1 = MagicMock()
        mock_version1.version = "18"
        mock_version1.aliases = ["production", "latest"]

        mock_version2 = MagicMock()
        mock_version2.version = "19"
        mock_version2.aliases = ["staging"]

        mlflow_loader.client.search_model_versions.return_value = [
            mock_version1,
            mock_version2,
        ]

        # Act
        aliases_map = mlflow_loader.list_all_aliases()

        # Assert
        expected_map = {"production": "18", "latest": "18", "staging": "19"}
        assert aliases_map == expected_map

    def test_exception_handling_returns_appropriate_defaults(self, mlflow_loader):
        """Test that methods handle exceptions gracefully."""
        # Arrange
        mlflow_loader.client.get_model_version_by_alias.side_effect = Exception(
            "MLflow error"
        )
        mlflow_loader.client.search_model_versions.side_effect = Exception(
            "Search failed"
        )
        mlflow_loader.client.get_model_version.side_effect = Exception(
            "Get model version failed"
        )

        # Act & Assert
        assert mlflow_loader.get_model_by_alias("production") is None
        assert mlflow_loader.list_all_aliases() == {}
        assert mlflow_loader.get_model_aliases("6") == []


class TestMLflowArtifactLoaderIntegration:
    """Integration test for complete workflow."""

    def test_model_promotion_workflow(self, mlflow_loader, capsys):
        """Test complete workflow: validation check -> production promotion."""
        # Arrange
        staging_version = MagicMock()
        staging_version.version = "7"
        staging_version.tags = {"validation_status": "approved"}
        staging_version.aliases = ["staging"]

        mlflow_loader.client.get_model_version_by_alias.side_effect = [
            staging_version,  # For validaton check
            None,  # For production promotion (no existing)
        ]
        mlflow_loader.client.get_model_version.return_value = staging_version

        # Act
        is_validated = mlflow_loader.check_manual_validation_status_by_alias("staging")
        assert is_validated is True

        mlflow_loader.promote_to_production("9")

        # Assert
        mlflow_loader.client.set_registered_model_alias.assert_called_with(
            name=MLFLOW_EXPERIMENT_NAME, alias="production", version="9"
        )

        captured = capsys.readouterr()
        assert "✅ Model version 9 promoted to production" in captured.out
