import subprocess
from unittest.mock import MagicMock, patch

import pytest
from services.deployment.base import DeploymentStrategy
from services.deployment.inference_deployment import InferenceContainerDeployment


class TestInferenceContainerDeployment:
    """Critical tests for InferenceContainerDeployment."""

    def test_inheritance_from_deployment_strategy(self):
        """Test that InferenceContainerDeployment properly inherits from DeploymentStrategy."""
        deployment = InferenceContainerDeployment()
        assert isinstance(deployment, DeploymentStrategy)
        assert hasattr(deployment, "deploy")

    @patch("subprocess.run")
    @patch("services.deployment.mlflow_deployment.MLflowArtifactLoader")
    def test_deploy_success(self, mock_loader_class, mock_subprocess):
        """Test successful deployment with MLflow promotion and container restart."""
        # Arrange - MLflow mock
        mock_loader = MagicMock()
        mock_loader.promote_to_production.return_value = None
        mock_loader_class.return_value = mock_loader

        # Arrange - subprocess mock
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Container restarted successfully"
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        deployment = InferenceContainerDeployment()
        model_version = "1.2.3"

        # Act
        result = deployment.deploy(model_version)

        # Assert
        assert result["status"] == "success"
        assert result["model_version"] == model_version
        assert result["container_restarted"] == "inference-container"
        assert model_version in result["message"]
        assert "deploy to inference container" in result["message"]

        # Verify MLflow operations
        mock_loader_class.assert_called_once()
        mock_loader.promote_to_production.assert_called_once_with(model_version)

        # Verify subprocess operations
        mock_subprocess.assert_called_once_with(
            ["docker", "compose", "restart", "inference-container"],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )

    @patch("subprocess.run")
    @patch("services.deployment.mlflow_deployment.MLflowArtifactLoader")
    def test_deploy_mlflow_failure(self, mock_loader_class, mock_subprocess):
        """Test deployment failure when MLflow promotion fails."""
        # Arrange - MLflow fails
        mock_loader = MagicMock()
        mock_loader.promote_to_production.side_effect = Exception(
            "MLflow connection failed"
        )
        mock_loader_class.return_value = mock_loader

        deployment = InferenceContainerDeployment()
        model_version = "1.2.3"

        # Act
        result = deployment.deploy(model_version)

        # Assert
        assert result["status"] == "failed"
        assert result["error"] == "MLflow connection failed"
        assert "container_restarted" in result
        assert result["container_restarted"] is False

        # MLflow should be called but subprocess should NOT be called
        mock_loader.promote_to_production.assert_called_once_with(model_version)
        mock_subprocess.assert_not_called()

    @patch("subprocess.run")
    @patch("services.deployment.mlflow_deployment.MLflowArtifactLoader")
    def test_deploy_container_restart_failure(self, mock_loader_class, mock_subprocess):
        """Test deployment failure when container restart fails."""
        # Arrange - MLflow succeeds
        mock_loader = MagicMock()
        mock_loader.promote_to_production.return_value = None
        mock_loader_class.return_value = mock_loader

        # Arrange - subprocess fails
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Container restart failed"
        mock_subprocess.return_value = mock_result

        deployment = InferenceContainerDeployment()
        model_version = "1.2.3"

        # Act
        result = deployment.deploy(model_version)

        # Assert
        assert result["status"] == "failed"
        assert result["error"] == "Container restart failed"
        assert result["container_restarted"] is False

        # Both MLflow and subprocess should be called
        mock_loader.promote_to_production.assert_called_once_with(model_version)
        mock_subprocess.assert_called_once()

    @patch("subprocess.run")
    @patch("services.deployment.mlflow_deployment.MLflowArtifactLoader")
    def test_deploy_subprocess_exception(self, mock_loader_class, mock_subprocess):
        """Test deployment failure when subprocess raises exception."""
        # Arrange - MLflow succeeds
        mock_loader = MagicMock()
        mock_loader_class.return_value = mock_loader

        # Arrange - subprocess raises exception
        mock_subprocess.side_effect = subprocess.TimeoutExpired(
            cmd=["docker", "compose", "restart", "inference-container"], timeout=60
        )

        deployment = InferenceContainerDeployment()
        model_version = "1.2.3"

        # Act
        result = deployment.deploy(model_version)

        # Assert
        assert result["status"] == "failed"
        assert "TimeoutExpired" in result["error"]
        assert "container_restarted" in result
        assert result["container_restarted"] is False

    @patch("services.deployment.mlflow_deployment.MLflowArtifactLoader")
    def test_deploy_loader_instantiation_failure(self, mock_loader_class):
        """Test deployment failure when MLflowArtifactLoader instantiation fails."""
        # Arrange
        mock_loader_class.side_effect = Exception("Failed to initialize MLflow client")

        deployment = InferenceContainerDeployment()
        model_version = "1.2.3"

        # Act
        result = deployment.deploy(model_version)

        # Assert
        assert result["status"] == "failed"
        assert result["error"] == "Failed to initialize MLflow client"

    def test_deploy_method_signature(self):
        """Test that deploy method has correct signature."""
        deployment = InferenceContainerDeployment()

        # Should be callable with model_version
        assert callable(deployment.deploy)

        # Check method signature
        import inspect

        sig = inspect.signature(deployment.deploy)
        assert "model_version" in sig.parameters
        assert sig.parameters["model_version"].annotation == str

    @patch("subprocess.run")
    @patch("services.deployment.mlflow_deployment.MLflowArtifactLoader")
    def test_subprocess_timeout_parameter(self, mock_loader_class, mock_subprocess):
        """Test that subprocess is called with correct timeout value."""
        # Arrange
        mock_loader = MagicMock()
        mock_loader_class.return_value = mock_loader
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result

        deployment = InferenceContainerDeployment()

        # Act
        deployment.deploy("1.0.0")

        # Assert
        call_args = mock_subprocess.call_args
        assert call_args[1]["timeout"] == 60  # 1 minute as specified

    @patch("subprocess.run")
    @patch("services.deployment.mlflow_deployment.MLflowArtifactLoader")
    def test_subprocess_command_structure(self, mock_loader_class, mock_subprocess):
        """Test that subprocess is called with correct Docker command."""
        # Arrange
        mock_loader = MagicMock()
        mock_loader_class.return_value = mock_loader
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result

        deployment = InferenceContainerDeployment()

        # Act
        deployment.deploy("1.0.0")

        # Assert
        expected_command = ["docker", "compose", "restart", "inference-container"]
        call_args = mock_subprocess.call_args
        assert call_args[0][0] == expected_command

    @patch("subprocess.run")
    @patch("services.deployment.mlflow_deployment.MLflowArtifactLoader")
    def test_return_dict_structure_success(self, mock_loader_class, mock_subprocess):
        """Test that success return dictionary has correct structure."""
        # Arrange
        mock_loader = MagicMock()
        mock_loader_class.return_value = mock_loader
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result

        deployment = InferenceContainerDeployment()
        result = deployment.deploy("1.0.0")

        # Assert required keys for success
        required_success_keys = {
            "status",
            "model_version",
            "container_restarted",
            "message",
        }
        assert set(result.keys()) == required_success_keys

        # Assert value types and specific values
        assert isinstance(result["status"], str)
        assert isinstance(result["model_version"], str)
        assert isinstance(result["container_restarted"], str)
        assert isinstance(result["message"], str)
        assert result["status"] == "success"
        assert result["container_restarted"] == "inference-container"

    @patch("subprocess.run")
    @patch("services.deployment.mlflow_deployment.MLflowArtifactLoader")
    def test_return_dict_structure_failure(self, mock_loader_class, mock_subprocess):
        """Test that failure return dictionary has correct structure."""
        # Arrange - force failure
        mock_loader_class.side_effect = Exception("Test error")

        deployment = InferenceContainerDeployment()
        result = deployment.deploy("1.0.0")

        # Assert required keys for failure
        required_failure_keys = {
            "status",
            "error",
            "container_restarted",
        }
        assert set(result.keys()) == required_failure_keys

        # Assert value types
        assert isinstance(result["status"], str)
        assert isinstance(result["error"], str)
        assert isinstance(result["container_restarted"], bool)
        assert result["status"] == "failed"
        assert result["container_restarted"] is False

    @patch("subprocess.run")
    @patch("services.deployment.mlflow_deployment.MLflowArtifactLoader")
    def test_deployment_workflow_order(self, mock_loader_class, mock_subprocess):
        """Test that MLflow promotion happens before container restart."""
        # Arrange
        mock_loader = MagicMock()
        mock_loader_class.return_value = mock_loader
        mock_result = MagicMock()
        mock_result.return_code = 0
        mock_subprocess.return_value = mock_result

        deployment = InferenceContainerDeployment()

        # Act
        deployment.deploy("1.0.0")

        # Assert order of operations
        mock_loader.promote_to_production.assert_called_once()
        mock_subprocess.assert_called_once()

        # If MLflow fails, subprocess should not be called (tested in other test)
        # This test just ensures both are called when successful

    @patch("subprocess.run")
    @patch("services.deployment.mlflow_deployment.MLflowArtifactLoader")
    def test_container_restarted_value_types(self, mock_loader_class, mock_subprocess):
        """Test that container_restarted has correct value types in different scenarios."""
        mock_loader = MagicMock()
        mock_loader_class.return_value = mock_loader
        deployment = InferenceContainerDeployment()

        # Success case - should be string with container name
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result

        result = deployment.deploy("1.0.0")
        assert isinstance(result["container_restarted"], str)
        assert result["container_restarted"] == "inference-container"

        # Failure case - should be boolean False
        mock_result.returncode = 1
        mock_result.stderr = "Error"
        mock_subprocess.return_value = mock_result

        result = deployment.deploy("1.0.0")
        assert isinstance(result["container_restarted"], bool)
        assert result["container_restarted"] is False
