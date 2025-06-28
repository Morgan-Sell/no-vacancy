from unittest.mock import MagicMock, patch
import pytest
from services.deployment.mlflow_deployment import MLflowDeployment
from services.deployment.training_deployment import TrainingContainerDeployment
from services.deployment.inference_deployment import InferenceContainerDeployment
from config import INFERENCE_CONTAINER, CDConfig, DeploymentMode
from services.cd_pipeline import CDPipeline


class TestCDPipeline:
    """Critical tests for CDPipeline coordinator"""

    def test_pipeline_initialization(self):
        # Arrange
        config = CDConfig.for_production_inference()

        # Act
        pipeline = CDPipeline(config)

        # Assert
        assert pipeline.config == config
        assert pipeline.validator is not None
        assert pipeline.deployment is not None

    def test_factory_inference_deployment_strategy(self):
        # Arrange
        config = CDConfig.for_production_inference()

        # Act
        pipeline = CDPipeline(config)

        # Assert
        assert isinstance(pipeline.deployment, InferenceContainerDeployment)

    def test_factory_creates_training_deployment_strategy(self):
        # Arrange
        config = CDConfig.for_automated_training()

        # Act
        pipeline = CDPipeline(config)

        # Assert
        assert isinstance(pipeline.deployment, TrainingContainerDeployment)

    def test_factory_creates_mlflow_deployment_strategy(self):
        # Arrange
        config = CDConfig.for_staging_mlflow()

        # Act
        pipeline = CDPipeline(config)

        # Assert
        assert isinstance(pipeline.deployment, MLflowDeployment)

    def test_factory_raises_error_for_unknown_deployment_mode(self):
        # Arrange
        config = CDConfig(
            target_environment="test",
            require_manual_validation=False,
            deployment_mode="EXPLODE_BOOM!",
        )

        # Act & Assert
        with pytest.raises((ValueError, TypeError)):
            CDPipeline(config)

    @patch("services.cd_pipeline.MLflowArtifactLoader")
    def test_training_deployment_success(self, mock_loader_class):
        # Arrange
        config = CDConfig.for_automated_training()
        pipeline = CDPipeline(config)

        # Mock deployment strategy
        mock_deployment = MagicMock()
        mock_deployment.deploy.return_value = {
            "status": "success",
            "action": "training_triggered",
        }
        pipeline.deployment = mock_deployment

        # Act
        result = pipeline.deploy_latest_model()

        # Assert
        assert result == "✅ Training completed successfully"
        mock_deployment.deploy.assert_called_once_with()  # No model version for training

    @patch("services.cd_pipeline.MLflowArtifactLoader")
    def test_training_deployment_failure(self, mock_loader_class):
        """Test failed training deployment workflow."""
        # Arrange
        config = CDConfig.for_automated_training()
        pipeline = CDPipeline(config)

        # Mock deployment strategy
        mock_deployment = MagicMock()
        mock_deployment.deploy.return_value = {
            "status": "failed",
            "error": "Training container failed to start",
        }
        pipeline.deployment = mock_deployment

        # Act
        result = pipeline.deploy_latest_model()

        # Assert
        assert result == "❌ Training failed: Training container failed to start"
        mock_deployment.deploy.assert_called_once_with()

    @patch("services.cd_pipeline.MLflowArtifactLoader")
    def test_inference_deployment_success(self, mock_loader_class):
        """Test successful inference deployment workflow."""
        # Arrange
        config = CDConfig.for_production_inference()
        pipeline = CDPipeline(config)

        # Mock MLflow loader
        mock_loader = MagicMock()
        mock_loader.get_artifact_metadata.return_value = {"version": "1.2.3"}
        mock_loader_class.return_value = mock_loader

        # Mock validator
        pipeline.validator = MagicMock()
        pipeline.validator.validate.return_value = True

        # Mock deployment strategy
        mock_deployment = MagicMock()
        mock_deployment.deploy.return_value = {
            "status": "success",
            "model_version": "1.2.3",
            "container_restarted": INFERENCE_CONTAINER,
        }
        pipeline.deployment = mock_deployment

        # Act
        result = pipeline.deploy_latest_model()

        # Assert
        assert "✅ Successfully deployed model version 1.2.3" in result
        assert f"{INFERENCE_CONTAINER} restarted" in result
        mock_deployment.deploy.assert_called_once_with("1.2.3")

    @patch("services.cd_pipeline.MLflowArtifactLoader")
    def test_inference_deployment_no_staging_model(self, mock_loader_class):
        """Test inference deployment when no model in Staging."""
        # Arrange
        config = CDConfig.for_production_inference()
        pipeline = CDPipeline(config)

        # Mock MLflow loader to return no staging model
        mock_loader = MagicMock()
        mock_loader.get_artifact_metadata.return_value = {}
        mock_loader_class.return_value = mock_loader

        # Act
        result = pipeline.deploy_latest_model()

        # Assert
        assert result == "❌ No manually validated models found in Staging"

    @patch("services.cd_pipeline.MLflowArtifactLoader")
    def test_inference_deployment_validation_failure(self, mock_loader_class):
        """Test inference deployment when validation fails."""
        # Arrange
        config = CDConfig.for_production_inference()
        pipeline = CDPipeline(config)

        # Mock MLflow loader
        mock_loader = MagicMock()
        mock_loader.get_artifact_metadata.return_value = {"version": "1.2.3"}
        mock_loader_class.return_value = mock_loader

        # Mock validator to fail
        pipeline.validator = MagicMock()
        pipeline.validator.validate.return_value = False

        # Act
        result = pipeline.deploy_latest_model()

        # Assert
        assert result == "❌ Model version 1.2.3 failed validation"

    @patch("services.cd_pipeline.MLflowArtifactLoader")
    def test_inference_deployment_deployment_failure(self, mock_loader_class):
        """Test inference deployment when deployment strategy fails."""
        # Arrange
        config = CDConfig.for_production_inference()
        pipeline = CDPipeline(config)

        # Mock MLflow loader
        mock_loader = MagicMock()
        mock_loader.get_artifact_metadata.return_value = {"version": "1.2.3"}
        mock_loader_class.return_value = mock_loader

        # Mock validator
        pipeline.validator = MagicMock()
        pipeline.validator.validate.return_value = True

        # Mock deployment strategy to fail
        mock_deployment = MagicMock()
        mock_deployment.deploy.return_value = {
            "status": "failed",
            "error": "Container restart failed",
        }
        pipeline.deployment = mock_deployment

        # Act
        result = pipeline.deploy_latest_model()

        # Assert
        assert result == "❌ Deployment failed: Container restart failed"

    @patch("services.cd_pipeline.MLflowArtifactLoader")
    def test_get_latest_validated_model_success(self, mock_loader_class):
        """Test _get_latest_validated_model returns version from Staging."""
        # Arrange
        config = CDConfig.for_production_inference()
        pipeline = CDPipeline(config)

        mock_loader = MagicMock()
        mock_loader.get_artifact_metadata.return_value = {"version": "2.1.0"}
        mock_loader_class.return_value = mock_loader

        # Act
        result = pipeline._get_latest_validated_model()

        # Assert
        assert result == "2.1.0"
        mock_loader.get_artifact_metadata.assert_called_once_with("Staging")

    @patch("services.cd_pipeline.MLflowArtifactLoader")
    def test_get_latest_validated_model_exception(self, mock_loader_class):
        """Test _get_latest_validated_model handles exceptions gracefully."""
        # Arrange
        config = CDConfig.for_production_inference()
        pipeline = CDPipeline(config)

        mock_loader = MagicMock()
        mock_loader.get_artifact_metadata.side_effect = Exception("MLflow error")
        mock_loader_class.return_value = mock_loader

        # Act
        result = pipeline._get_latest_validated_model()

        # Assert
        assert result is None

    def test_validate_model_with_manual_validation_required(self):
        """Test _validate_model when manual validation is required."""
        # Arrange
        config = CDConfig.for_production_inference()  # Requires manual validation
        pipeline = CDPipeline(config)

        # Mock validator
        pipeline.validator = MagicMock()
        pipeline.validator.validate.return_value = True

        # Act & Assert
        with pytest.raises(AttributeError):
            pipeline._validate_model("1.0.0")

    def test_validate_model_without_manual_validation(self):
        """Test _validate_model when manual validation is not required."""
        # Arrange
        config = CDConfig.for_staging_mlflow()  # No manual validation required
        pipeline = CDPipeline(config)

        # Act & Assert
        with pytest.raises(AttributeError):
            result = pipeline._validate_model("1.0.0")

    def test_container_restart_message_formatting(self):
        """Test different container restart message formats."""
        # Arrange
        config = CDConfig.for_production_inference()
        pipeline = CDPipeline(config)

        # Mock successful deployment with string container name
        mock_deployment = MagicMock()
        mock_deployment.deploy.return_value = {
            "status": "success",
            "model_version": "1.0.0",
            "container_restarted": INFERENCE_CONTAINER,
        }
        pipeline.deployment = mock_deployment

        # Mock other dependencies
        pipeline._get_latest_validated_model = MagicMock(return_value="1.0.0")
        pipeline._validate_model = MagicMock(return_value=True)

        # Act
        result = pipeline.deploy_latest_model()

        # Assert
        assert "inference-container restarted" in result

    def test_mlflow_only_deployment(self):
        """Test MLflow-only deployment workflow."""
        # Arrange
        config = CDConfig.for_staging_mlflow()
        pipeline = CDPipeline(config)

        # Mock deployment strategy
        mock_deployment = MagicMock()
        mock_deployment.deploy.return_value = {
            "status": "success",
            "model_version": "1.0.0",
            "container_restarted": False,
        }
        pipeline.deployment = mock_deployment

        # Mock other dependencies
        pipeline._get_latest_validated_model = MagicMock(return_value="1.0.0")
        pipeline._validate_model = MagicMock(return_value=True)

        # Act
        result = pipeline.deploy_latest_model()

        # Assert
        assert "✅ Successfully deployed model version 1.0.0" in result
        assert "container" not in result.lower()  # No container restart message
