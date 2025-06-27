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

    def _create_deployment_strategy(self):
        """Factory method to create appropriate deployment strategy."""
        if self.config.deployment_mode == DeploymentMode.INFERENCE_CONTAINER_RESTART:
            return InferenceContainerDeployment()
        elif self.config.deployment_mode == DeploymentMode.TRAINING_CONTAINER_RUN:
            return TrainingContainerDeployment()
        elif self.config.deployment_mode == DeploymentMode.MLFLOW_ONLY:
            return MLflowDeployment()
        else:
            raise ValueError(f"Unknown deployment mode: {self.config.deployment_mode}")

    def deploy_latest_model(self) -> str:
        """
        Main deployment pipeline for 3-container architecture.
        Returns: status message
        """
        # Training workflows do not require a specific model version
        if self.config.deployment_mode == DeploymentMode.TRAINING_CONTAINER_RUN:
            deployment_result = self.deployment.deploy()
            if deployment_result["status"] == "success":
                return "✅ Training completed successfully"
            else:
                return f"❌ Training failed: {deployment_result.get('error', 'Unknown error')}"

        # Inference deployment: Get the latest validated model
        model_version = self._get_latest_validated_model()
        if not model_version:
            return "❌ No manually validated models found in Staging"

        # Validate using Strategy pattern
        if not self._validate_model(model_version):
            return f"❌ Model version {model_version} failed validation"

        # Deploy using Strategy pattern
        deployment_result = self.deployment.deploy(model_version)

        if deployment_result["status"] == "success":
            container_msg = ""
            if deployment_result.get("container_restarted"):
                if isinstance(deployment_result["container_restarted"], str):
                    container_msg = (
                        f" ({deployment_result['container_restarted']} restarted)"
                    )
                else:
                    container_msg = " (container restarted)"
            return f"✅ Successfully deployed model version {model_version} to {self.config.target_environment}{container_msg}"
        else:
            return f"❌ Deployment failed: {deployment_result.get('error', 'Unknown error')}"

    def _get_latest_validated_model(self) -> Union[str, None]:
        """Get the latest model version that's in the Staging stage"""
        loader = MLflowArtifactLoader()
        try:
            metadata = loader.get_artifact_metadata("Staging")
            return metadata.get("version")
        except Exception:
            return None

    def _validate_model(self, model_version: str) -> bool:
        if self.config.require_manual_validation:
            return self.validator.validate(model_version)
        return True
