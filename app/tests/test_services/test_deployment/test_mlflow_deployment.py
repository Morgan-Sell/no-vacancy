from unittest.mock import MagicMock, patch

import pytest
from services.deployment.base import DeploymentStrategy
from services.deployment.mlflow_deployment import MLflowDeployment


class TestMLflowDeployment:
    """Critical tests for MLflowDeployment."""

    def test_inheritance_from_deployment_strategy(self):
        deployment = MLflowDeployment()
        assert isinstance(deployment, DeploymentStrategy)
        assert hasattr(deployment, "deploy")

    # @patch creates mock_loader_class
    @patch("services.deployment.mlflow_deployment.MLflowArtifactLoader")
    def test_deploy_success(self, mock_loader_class):
        # Arrange
        mock_loader = MagicMock()
        mock_loader.promote_to_production.return_value = None
        mock_loader_class.return_value = mock_loader

        deployment = MLflowDeployment()
        model_version = "9.9.9"

        # Act
        result = deployment.deploy(model_version)

        # Assert
        assert result["status"] == "success"
        assert result["model_version"] == model_version
        assert result["container_restarted"] is False
        assert "promoted to Production stage" in result["message"]
        assert model_version is result["message"]

    @patch("services.deployment.mlflow_deployment.MLflowArtifactLoader")
    def test_deploy_mlflow_exception(self, mock_loader_class):
        """Test deployment failure when MLflow operations fail."""
        # Arrange
        mock_loader = MagicMock()
        mock_loader.promote_to_production.side_effect = Exception(
            "MLflow connection failed"
        )
        mock_loader_class.return_value = mock_loader

        deployment = MLflowDeployment()
        model_version = "1.2.3"

        # Act
        result = deployment.deploy(model_version)

        # Assert
        assert result["status"] == "failed"
        assert result["error"] == "MLflow connection failed"
        assert "model_version" not in result  # Should not be present on failure
        assert "message" not in result  # Should not be present on failure

    @patch("services.deployment.mlflow_deployment.MLflowArtifactLoader")
    def test_deploy_loader_instantiation_failure(self, mock_loader_class):
        """Test deployment failure when MLflowArtifactLoader instantiation fails."""
        # Arrange
        mock_loader_class.side_effect = Exception("Failed to initialize MLflow client")

        deployment = MLflowDeployment()
        model_version = "1.2.3"

        # Act
        result = deployment.deploy(model_version)

        # Assert
        assert result["status"] == "failed"
        assert result["error"] == "Failed to initialize MLflow client"

    def test_deploy_method_signature(self):
        """Test that deploy method has correct signature."""
        deployment = MLflowDeployment()

        # Should be callable with model_version
        assert callable(deployment.deploy)

        # Check method signature
        import inspect

        sig = inspect.signature(deployment.deploy)
        assert "model_version" in sig.parameters
        assert sig.parameters["model_version"].annotation == str

    @patch("services.deployment.mlflow_deployment.MLflowArtifactLoader")
    def test_deploy_with_different_model_versions(self, mock_loader_class):
        """Test deployment works with various model version formats."""
        # Arrange
        mock_loader = MagicMock()
        mock_loader.promote_to_production.return_value = None
        mock_loader_class.return_value = mock_loader

        deployment = MLflowDeployment()

        # Test different version formats
        test_versions = ["1", "1.0", "1.0.0", "v1.2.3", "latest"]

        for version in test_versions:
            # Act
            result = deployment.deploy(version)

            # Assert
            assert result["status"] == "success"
            assert result["model_version"] == version
            assert version in result["message"]

    @patch("services.deployment.mlflow_deployment.MLflowArtifactLoader")
    def test_container_restarted_always_false(self, mock_loader_class):
        """Test that container_restarted is always False for MLflow-only deployment."""
        # Arrange
        mock_loader = MagicMock()
        mock_loader.promote_to_production.return_value = None
        mock_loader_class.return_value = mock_loader

        deployment = MLflowDeployment()

        # Act
        result = deployment.deploy("1.0.0")

        # Assert
        assert result["container_restarted"] is False
        # This is a key differentiator from InferenceContainerDeployment

    @patch("services.deployment.mlflow_deployment.MLflowArtifactLoader")
    def test_return_dict_structure_success(self, mock_loader_class):
        """Test that success return dictionary has correct structure."""
        # Arrange
        mock_loader = MagicMock()
        mock_loader_class.return_value = mock_loader

        deployment = MLflowDeployment()
        result = deployment.deploy("1.0.0")

        # Assert required keys for success
        required_success_keys = {
            "status",
            "model_version",
            "container_restarted",
            "message",
        }
        assert set(result.keys()) == required_success_keys

        # Assert value types
        assert isinstance(result["status"], str)
        assert isinstance(result["model_version"], str)
        assert isinstance(result["container_restarted"], bool)
        assert isinstance(result["message"], str)

    @patch("services.deployment.mlflow_deployment.MLflowArtifactLoader")
    def test_return_dict_structure_failure(self, mock_loader_class):
        """Test that failure return dictionary has correct structure."""
        # Arrange
        mock_loader_class.side_effect = Exception("Test error")

        deployment = MLflowDeployment()
        result = deployment.deploy("1.0.0")

        # Assert required keys for failure
        required_failure_keys = {"status", "error"}
        assert set(result.keys()) == required_failure_keys

        # Assert value types
        assert isinstance(result["status"], str)
        assert isinstance(result["error"], str)
        assert result["status"] == "failed"

    @patch("services.deployment.mlflow_deployment.MLflowArtifactLoader")
    def test_mlflow_loader_called_correctly(self, mock_loader_class):
        """Test that MLflowArtifactLoader is instantiated and called correctly."""
        # Arrange
        mock_loader = MagicMock()
        mock_loader_class.return_value = mock_loader

        deployment = MLflowDeployment()
        model_version = "2.1.0"

        # Act
        deployment.deploy(model_version)

        # Assert
        # Loader should be instantiated once
        mock_loader_class.assert_called_once_with()

        # promote_to_production should be called once with correct version
        mock_loader.promote_to_production.assert_called_once_with(model_version)

    def test_class_docstring_describes_purpose(self):
        """Test that class has appropriate docstring describing MLflow-only deployment."""
        deployment = MLflowDeployment()
        docstring = deployment.__class__.__doc__

        assert docstring is not None
        assert "MLflow" in docstring
        # Should mention it handles MLflow workflow, not container management
        assert any(
            keyword in docstring.lower()
            for keyword in ["mlflow", "staging", "production"]
        )

    @patch("services.deployment.mlflow_deployment.MLflowArtifactLoader")
    def test_deploy_empty_string_model_version(self, mock_loader_class):
        """Test deployment behavior with edge case model version."""
        # Arrange
        mock_loader = MagicMock()
        mock_loader_class.return_value = mock_loader

        deployment = MLflowDeployment()

        # Act
        result = deployment.deploy("")

        # Assert - Should still work (MLflow might handle empty versions)
        assert result["status"] == "success"
        assert result["model_version"] == ""
        mock_loader.promote_to_production.assert_called_once_with("")
