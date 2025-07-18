import os
from typing import Any, Dict, Tuple, Union

import joblib
import mlflow
from services import MLFLOW_EXPERIMENT_NAME, MLFLOW_PROCESSOR_PATH, MLFLOW_TRACKING_URI


class MLflowArtifactLoader:
    """
    Centralized MLflow loader for models, processors, and metadata.
    """

    def __init__(self):
        """Initialize MLflow connection."""
        # Use os.getenv to improve testability and flexibility for CLI overrides
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI))
        self.client = mlflow.MlflowClient()

    def load_pipeline_artifacts_by_stage(
        self, stage: str = "Production"
    ) -> Tuple[Any, Any, str]:
        """Load model and processor artifacts from MLflow by stage."""
        # Get latest version in stage
        versions = self.client.get_latest_versions(
            MLFLOW_EXPERIMENT_NAME, stages=[stage]
        )
        if not versions:
            raise RuntimeError(
                f"No model version found for stage '{stage}' in experiment '{MLFLOW_EXPERIMENT_NAME}'"
            )

        run_id = versions[0].run_id
        model_version = versions[0].version

        # Load model artifact
        model_uri = f"models:/{MLFLOW_EXPERIMENT_NAME}/{stage}"
        pipeline = mlflow.sklearn.load_model(model_uri=model_uri)

        # Load processor artifact
        local_processor_path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path=MLFLOW_PROCESSOR_PATH
        )
        processor = joblib.load(local_processor_path)

        return pipeline, processor, model_version

    def load_pipeline_artifacts_by_version(self, model_version: str) -> Tuple[Any, Any]:
        """Load model and processor artifacts by specific version."""
        # Get run info for the specified version
        model_version_obj = self.client.get_model_version(
            name=MLFLOW_EXPERIMENT_NAME, version=model_version
        )
        run_id = model_version_obj.run_id

        # Load model artifact
        model_uri = f"models:/{MLFLOW_EXPERIMENT_NAME}/{model_version}"
        pipeline = mlflow.sklearn.load_model(model_uri=model_uri)

        # Load processor artifact
        local_processor_path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path=MLFLOW_PROCESSOR_PATH
        )
        processor = joblib.load(local_processor_path)

        return pipeline, processor

    def get_artifact_metadata(self, stage: str) -> Dict[str, Any]:
        """Get metadata about the artifacts in a specific stage."""
        versions = self.client.get_latest_versions(
            MLFLOW_EXPERIMENT_NAME, stages=[stage]
        )
        if not versions:
            return {}

        version = versions[0]
        run = self.client.get_run(version.run_id)

        return {
            "version": version.version,
            "run_id": version.run_id,
            "stage": version.current_stage,
            "metrics": run.data.metrics,
            "artifacts": run.info.artifact_uri,
            "created_at": version.creation_timestamp,
        }

    def check_manual_validation_status(self, model_version: str) -> bool:
        """Check if the model was manually validated in MLflow."""
        try:
            model_details = self.client.get_model_version(
                name=MLFLOW_EXPERIMENT_NAME, version=model_version
            )

            tags = model_details.tags
            is_manually_validated = any(
                [
                    tags.get("manual_validation") == "approved",
                    tags.get("data_scientist_approved") == "true",
                    tags.get("validated") == "true",
                    tags.get("ready_for_production") == "true",
                ]
            )

            is_in_staging = model_details.current_stage == "Staging"

            return is_manually_validated and is_in_staging

        except Exception:
            return False

    def promote_to_production(self, model_version: str) -> None:
        """Promote model to production stage - triggers prediction container update."""
        self.client.transition_model_version_stage(
            name=MLFLOW_EXPERIMENT_NAME,
            version=model_version,
            stage="Production",
            archive_existing_versions=True,
        )

    def get_latest_production_model(self) -> Union[str, None]:
        """Get the latest model version in Production stage."""
        try:
            metadata = self.get_artifact_metadata(stage="Production")
            return metadata.get("version", "No production model found")

        except Exception:
            return None
