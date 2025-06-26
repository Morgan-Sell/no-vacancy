from services.deployment.base import DeploymentStrategy
from services.mlflow_utils import MLflowArtifactLoader


class MLflowDeployment(DeploymentStrategy):
    """
    Handles MLflow deployment workflow, not container management. Examples:
        - Promotes models from staging to production
        - Updates model metadata and tages
        - Changes which model version has "Production" status

    MLflow container doesn't require restarting because the MLflow server is stateless.
    """

    def deploy(self, model_version: str) -> dict:
        """
        Deploy by promoting model in MLflow only.
        Assumes inference container requests latest Production model from MLflow.
        """
        try:
            loader = MLflowArtifactLoader()
            loader.promote_to_production(model_version)

            return {
                "status": "success",
                "model_version": model_version,
                "container_restarted": False,
                "message": f"Model version {model_version} promoted to Production stage.",
            }

        except Exception as e:
            return {"status": "failed", "error": str(e)}
