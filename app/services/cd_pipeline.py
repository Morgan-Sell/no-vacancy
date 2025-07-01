from typing import Union

from config import CDConfig, DeploymentMode
from services.deployment.inference_deployment import InferenceContainerDeployment
from services.deployment.mlflow_deployment import MLflowDeployment
from services.deployment.training_deployment import TrainingContainerDeployment
from services.mlflow_utils import MLflowArtifactLoader
from services.validation.manual_validator import ManualValidator


class CDPipeline:
    """
    CD pipeline coordinator for 3 containers.
    Uses Factory pattern to choose appropriate deployment strategy.
    """

    def __init__(self, config: CDConfig):
        self.config = config

        # Strategy pattern - validator
        self.validator = ManualValidator()

        # Factory Pattern - Deployment strategy bsed on config
        self.deployment = self._create_deployment_strategy()

        print(f"🔍 DEBUG: CDPipeline initialized with mode: {config.deployment_mode}")
        print(f"🔍 DEBUG: Target environment: {config.target_environment}")
        print(
            f"🔍 DEBUG: Manual validation required: {config.require_manual_validation}"
        )

    def _create_deployment_strategy(self):
        """Factory method to create appropriate deployment strategy."""
        if self.config.deployment_mode == DeploymentMode.INFERENCE_CONTAINER_RESTART:
            print("🔍 DEBUG: Creating InferenceContainerDeployment strategy")
            return InferenceContainerDeployment()
        elif self.config.deployment_mode == DeploymentMode.TRAINING_CONTAINER_RUN:
            print("🔍 DEBUG: Creating TrainingContainerDeployment strategy")
            return TrainingContainerDeployment()
        elif self.config.deployment_mode == DeploymentMode.MLFLOW_ONLY:
            print("🔍 DEBUG: Creating MLflowDeployment strategy")
            return MLflowDeployment()
        else:
            raise ValueError(f"Unknown deployment mode: {self.config.deployment_mode}")

    def deploy_latest_model(self) -> str:
        """
        Main deployment pipeline for 3-container architecture.
        Returns: status message
        """
        print(f"🚀 Starting deployment with mode: {self.config.deployment_mode}")

        # Training workflows do not require a specific model version
        if self.config.deployment_mode == DeploymentMode.TRAINING_CONTAINER_RUN:
            return self._handle_training_deployment()

        # Inference deployment workflow
        return self._handle_inference_deployment()

    def _handle_training_deployment(self) -> str:
        """Handle training container deployment workflow."""
        print("🎯 Executing training workflow...")
        deployment_result = self.deployment.deploy()

        print(f"🔍 DEBUG: Deployment result: {deployment_result}")

        if deployment_result["status"] == "success":
            return "✅ Training completed successfully"

        # Build detailed error message
        return self._build_training_error_message(deployment_result)

    def _build_training_error_message(self, deployment_result: dict) -> str:
        """Build a detailed error message for training failures."""
        error_details = [
            f"Status: {deployment_result.get('status', 'unknown')}",
            f"Error: {deployment_result.get('error', 'Unknown error')}",
        ]

        if deployment_result.get("return_code"):
            error_details.append(f"Exit code: {deployment_result['return_code']}")

        if deployment_result.get("stdout"):
            error_details.append(f"STDOUT: {deployment_result['stdout'][:500]}...")

        if deployment_result.get("stderr"):
            error_details.append(f"STDERR: {deployment_result['stderr'][:500]}...")

        full_error = " | ".join(error_details)
        return f"❌ Training failed: {full_error}"

    def _handle_inference_deployment(self) -> str:
        """Handle inference container deployment workflow."""
        # Get the latest validated model
        model_version = self._get_latest_validated_model()
        if not model_version:
            return "❌ No manually validated models found in Staging"

        # Validate using Strategy pattern
        if not self._validate_model(model_version):
            return f"❌ Model version {model_version} failed validation"

        # Deploy using Strategy pattern
        deployment_result = self.deployment.deploy(model_version)

        if deployment_result["status"] == "success":
            container_msg = self._get_container_restart_message(deployment_result)
            return f"✅ Successfully deployed model version {model_version} to {self.config.target_environment}{container_msg}"

        return (
            f"❌ Deployment failed: {deployment_result.get('error', 'Unknown error')}"
        )

    def _get_container_restart_message(self, deployment_result: dict) -> str:
        """Generate container restart message from deployment result."""
        if not deployment_result.get("container_restarted"):
            return ""

        if isinstance(deployment_result["container_restarted"], str):
            return f" ({deployment_result['container_restarted']} restarted)"

        return " (container restarted)"

    def _get_latest_validated_model(self) -> Union[str, None]:
        """Get the latest model version that's in the Staging stage"""
        loader = MLflowArtifactLoader()
        try:
            metadata = loader.get_artifact_metadata("Staging")
            return metadata.get("version")
        except Exception:
            return None

    def _validate_model(self, model_version: str) -> bool:
        # If manual validation is required, validate
        if self.config.require_manual_validation:
            return self.validator.validate(model_version)
        # If validation isn't required, return True
        return True
