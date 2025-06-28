import subprocess

from config import (
    DOCKER_COMPOSE_RESTART_CMD,
    INFERENCE_CONTAINER,
    INFERENCE_DEPLOYMENT_TIMEOUT,
)
from services.deployment.base import DeploymentStrategy
from services.mlflow_utils import MLflowArtifactLoader


class InferenceContainerDeployment(DeploymentStrategy):
    """
    Deployment strategy for dedicated inference container.
    Promotes model in MLflow and restarts only the inference container.
    """

    def deploy(self, model_version: str) -> dict:
        """
        Deploy by promoting model and restarting inference container.
        Training container is unaffected.
        """
        try:
            # Promote modle in MLflow first
            loader = MLflowArtifactLoader()
            loader.promote_to_production(model_version)

            # Restart only the inference container
            result = subprocess.run(
                DOCKER_COMPOSE_RESTART_CMD + [INFERENCE_CONTAINER],
                capture_output=True,
                text=True,
                timeout=INFERENCE_DEPLOYMENT_TIMEOUT,
                check=True,
            )

            if result.returncode == 0:
                return {
                    "status": "success",
                    "model_version": model_version,
                    "container_restarted": "inference-container",
                    "message": f"Model {model_version} deploy to inference container",
                }

            else:
                return {
                    "status": "failed",
                    "error": result.stderr,
                    "container_restarted": False,
                }

        except Exception as e:
            return {"status": "failed", "error": str(e), "container_restarted": False}
