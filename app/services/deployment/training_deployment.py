import subprocess

from config import (
    DOCKER_COMPOSE_RUN_CMD,
    TRAINING_CONTAINER,
    TRAINING_DEPLOYMENT_TIMEOUT,
)
from services.deployment.base import DeploymentStrategy


class TrainingContainerDeployment(DeploymentStrategy):
    """
    Deployment strategy that triggers training in dedicated training container.
    Used for automated retraining workflows.
    """

    def deploy(self, model_version: str) -> dict:
        """
        Deploy by trigering training in training container.
        Used for scheduled retraiing or data drift scenarios.
        """
        try:
            # Run training container (it will exit when training completes)
            result = subprocess.run(
                DOCKER_COMPOSE_RUN_CMD + [TRAINING_CONTAINER],
                capture_output=True,
                text=True,
                timeout=TRAINING_DEPLOYMENT_TIMEOUT,
                check=True,  # Raise exception if command fails
            )  # 1 hour timeeout

            if result.returncode == 0:
                return {
                    "status": "success",
                    "action": "training_triggered",
                    "message": "Training completed successfully.",
                }

            else:
                return {
                    "status": "failed",
                    "error": result.stderr,
                    "action": "training_failed",
                }

        except subprocess.TimeoutExpired:
            return {"status": "failed", "error": "Training timed out after 1 hour."}

        except Exception as e:
            return {"status": "failed", "error": str(e)}
