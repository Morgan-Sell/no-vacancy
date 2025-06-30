from unittest.mock import MagicMock, patch

from config import TRAINING_DEPLOYMENT_TIMEOUT
from services.deployment.base import DeploymentStrategy
from services.deployment.training_deployment import TrainingContainerDeployment


class TestTrainingContainerDeployment:
    """Critical tests for TrainingContainerDeployment."""

    def test_inheritance_from_deployment_strategy(self):
        deployment = TrainingContainerDeployment()
        assert isinstance(deployment, DeploymentStrategy)
        assert hasattr(deployment, "deploy")

    # mock_subprocess is created by @patch
    @patch("subprocess.run")
    def test_deploy_success(self, mock_subprocess):
        # Arrange
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Training completed successfully"
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        deployment = TrainingContainerDeployment()

        # Act
        result = deployment.deploy("1.2.3")

        # Assert
        assert result["status"] == "success"
        assert result["action"] == "training_triggered"
        assert "Training completed successfully" in result["message"]

        # Verify subprocess.run was called with correct parameters
        mock_subprocess.assert_called_once_with(
            ["docker", "compose", "run", "--rm", "training-container"],
            capture_output=True,
            text=True,
            timeout=TRAINING_DEPLOYMENT_TIMEOUT,
            check=True,
        )

    @patch("subprocess.run")
    def test_deploy_failure_non_zero_returncode(self, mock_subprocess):
        # Arrange
        mock_result = MagicMock()
        mock_result.return_code = 1
        mock_result.stdout = ""
        mock_result.stderr = "Docker container failed to start"
        mock_subprocess.return_value = mock_result

        deployment = TrainingContainerDeployment()

        # Act
        result = deployment.deploy("3.6.9")

        # Assert
        assert result["status"] == "failed"
        assert result["action"] == "training_failed"
        assert result["error"] == "Docker container failed to start"

    @patch("subprocess.run")
    def test_deploy_general_exception(self, mock_subprocess):
        """Test deployment failure due to general exception."""
        # Arrange
        mock_subprocess.side_effect = Exception("Unexpected error occurred")

        deployment = TrainingContainerDeployment()

        # Act
        result = deployment.deploy("1.0.0")

        # Assert
        assert result["status"] == "failed"
        assert result["error"] == "Unexpected error occurred"

    def test_deploy_method_signature(self):
        """Test that deploy method accepts model_version parameter."""
        deployment = TrainingContainerDeployment()

        # Should be callable with model_version
        assert callable(deployment.deploy)

        # Check method signature accepts model_version parameter
        import inspect

        sig = inspect.signature(deployment.deploy)
        assert "model_version" in sig.parameters

    @patch("subprocess.run")
    def test_deploy_with_none_model_version(self, mock_subprocess):
        """Test deployment works with None model_version (training doesn't need specific version)."""
        # Arrange
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Training completed"
        mock_subprocess.return_value = mock_result

        deployment = TrainingContainerDeployment()

        # Act
        result = deployment.deploy(None)

        # Assert
        assert result["status"] == "success"
        # Verify the docker command doesn't include model version (training generates new models)
        mock_subprocess.assert_called_once_with(
            ["docker", "compose", "run", "--rm", "training-container"],
            capture_output=True,
            text=True,
            timeout=TRAINING_DEPLOYMENT_TIMEOUT,
            check=True,
        )

    @patch("subprocess.run")
    def test_subprocess_called_with_correct_timeout(self, mock_subprocess):
        """Test that subprocess is called with the correct timeout value."""
        # Arrange
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result

        deployment = TrainingContainerDeployment()

        # Act
        deployment.deploy("1.0.0")

        # Assert
        call_args = mock_subprocess.call_args
        assert call_args[1]["timeout"] == TRAINING_DEPLOYMENT_TIMEOUT

    @patch("subprocess.run")
    def test_return_dict_structure(self, mock_subprocess):
        """Test that return dictionary has correct structure for both success and failure."""
        # Test success case
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result

        deployment = TrainingContainerDeployment()
        result = deployment.deploy("1.0.0")

        # Success case should have these keys
        required_success_keys = {"status", "action", "message"}
        assert set(result.keys()) >= required_success_keys

        # Test failure case
        mock_subprocess.side_effect = Exception("Test error")
        result = deployment.deploy("1.0.0")

        # Failure case should have these keys
        required_failure_keys = {"status", "error"}
        assert set(result.keys()) >= required_failure_keys
