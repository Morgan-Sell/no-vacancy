from config import CDConfig, DeploymentMode
from services.deployment.inference_deployment import InferenceContainerDeployment
from services.deployment.mlflow_deployment import MLflowDeployment
from services.deployment.training_deployment import TrainingContainerDeployment
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
            return ValueError(f"Unknown deployment mode: {self.config.deployment_mode}")
