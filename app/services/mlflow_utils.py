import os
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import mlflow
from config import get_logger
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
        self.logger = get_logger(logger_name=__name__)

    def load_pipeline_artifacts_by_alias(
        self, alias: str = "production"
    ) -> Tuple[Any, Any, str]:
        """Load model and processor artifacts from MLflow by alias."""
        try:
            # Get model version by alias
            model_version = self.client.get_model_version_by_alias(
                name=MLFLOW_EXPERIMENT_NAME, alias=alias
            )

            if not model_version:
                raise RuntimeError(
                    f"No model version found with alias '{alias}' in experiment '{MLFLOW_EXPERIMENT_NAME}'"
                )

            run_id = model_version.run_id
            version = model_version.version

            # Load model artifact
            model_uri = f"models:/{MLFLOW_EXPERIMENT_NAME}@{alias}"
            pipeline = mlflow.sklearn.load_model(model_uri=model_uri)

            # Load processor artifact
            local_processor_path = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path=MLFLOW_PROCESSOR_PATH
            )
            processor = joblib.load(local_processor_path)

            return pipeline, processor, version

        except Exception as e:
            raise RuntimeError(
                f"Failed to load artifacts by alias '{alias}': {e}"
            ) from e

    def get_model_by_alias(self, alias: str) -> Optional[Any]:
        """Get model version by alias."""
        try:
            return self.client.get_model_version_by_alias(
                name=MLFLOW_EXPERIMENT_NAME, alias=alias
            )

        except Exception:
            return None

    def set_model_alias(self, version: str, alias: str) -> None:
        """Set alias for a model version."""
        try:
            self.client.set_registered_model_alias(
                name=MLFLOW_EXPERIMENT_NAME, alias=alias, version=version
            )
            print(f"✅ Set alias '{alias}' for model version {version}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to set alias '{alias}' for model version {version}: {e}"
            ) from e

    def delete_model_alias(self, alias: str) -> None:
        """Delete an alias from the model."""
        try:
            self.client.delete_registered_model_alias(
                name=MLFLOW_EXPERIMENT_NAME, alias=alias
            )
            print(f"✅ Deleted alias '{alias}'")
        except Exception as e:
            # Don't raise error if alias doesn't exist
            print(f"Warning: Could not delete alias '{alias}': {e}")

    def get_model_aliases(self, version: str) -> List[str]:
        """Get all aliases for a specific model version."""
        try:
            model_version = self.client.get_model_version(
                name=MLFLOW_EXPERIMENT_NAME, version=version
            )
            return model_version.aliases if hasattr(model_version, "aliases") else []
        except Exception:
            return []

    def promote_to_production(self, model_version: str) -> None:
        """Promote model to production using aliases."""
        try:
            # Remove current production alias if it exists
            current_prod = self.get_model_by_alias("production")
            if current_prod:
                print(f"Removing production alias from version {current_prod.version}")
                self.delete_model_alias("production")

            # Set new production alias
            self.set_model_alias(model_version, "production")
            print(f"✅ Model version {model_version} promoted to production")

        except Exception as e:
            raise RuntimeError(
                f"Failed to promote model version {model_version} to production: {e}"
            ) from e

    def promote_to_staging(self, model_version: str) -> None:
        """Promote model to staging using aliases."""
        try:
            # Remove current staging alias if it exists
            current_staging = self.get_model_by_alias("staging")
            if current_staging:
                print(f"Removing staging alias from version {current_staging.version}")
                self.delete_model_alias("staging")

            # Set new staging alias
            self.set_model_alias(model_version, "staging")
            print(f"✅ Model version {model_version} promoted to staging")

        except Exception as e:
            raise RuntimeError(
                f"Failed to promote model version {model_version} to staging: {e}"
            ) from e

    def get_artifact_metadata_by_alias(self, alias: str) -> Dict[str, Any]:
        """Get metadata about the artifacts with a specific alias."""
        model_version = self.get_model_by_alias(alias)
        if not model_version:
            return {}

        try:
            run = self.client.get_run(model_version.run_id)

            return {
                "version": model_version.version,
                "run_id": model_version.run_id,
                "alias": alias,
                "aliases": self.get_model_aliases(model_version.version),
                "metrics": run.data.metrics,
                "artifacts": run.info.artifact_uri,
                "created_at": model_version.creation_timestamp,
                "description": model_version.description,
                "tags": model_version.tags,
            }

        except Exception as e:
            print(
                f"⚠️ Warning: Could not retrieve metadata for {model_version.run_id} (alias: '{alias}')"
            )
            return {
                "version": model_version.version,
                "alias": alias,
                "run_id": model_version.run_id,
                "error": f"Run data not accessible: {str(e)}",
                "partial_data": True,
            }

    def check_manual_validation_status_by_alias(self, alias: str = "staging") -> bool:
        """Check if the model with alias was manually validated."""
        try:
            model_version = self.get_model_by_alias(alias)
            if not model_version:
                return False

            tags = model_version.tags
            is_manually_validated = any(
                [
                    # Industry standard MLflow tags
                    tags.get("validation_status") == "approved",
                    tags.get("model_approval") == "approved",
                    tags.get("deployment_approval") == "approved",
                    # Common MLOps platform tags
                    tags.get("model_quality") == "passed",
                    tags.get("performance_validation") == "passed",
                    # Azure ML / AWS SageMaker style
                    tags.get("ModelApprovalStatus") == "Approved",
                    tags.get("deployment_approved") == "true",
                ]
            )

            # Check if it has staging alias
            aliases = self.get_model_aliases(model_version.version)
            has_staging_alias = "staging" in aliases

            return is_manually_validated and has_staging_alias

        except Exception:
            return False

    def get_latest_production_model(self) -> Union[str, None]:
        """Get the latest model version with production alias."""
        try:
            metadata = self.get_artifact_metadata_by_alias("production")
            return metadata.get("version", None)
        except Exception:
            return None

    # ---------------------
    # UTILITY METHODS
    # ---------------------

    def load_pipeline_artifacts_by_version(self, model_version: str) -> Tuple[Any, Any]:
        """Load model and processor artifacts by specific version (still valid)."""
        try:
            # Get run info for the specified version
            model_version_obj = self.client.get_model_version(
                name=MLFLOW_EXPERIMENT_NAME, version=model_version
            )
            run_id = model_version_obj.run_id

            # Load model artifact using version number
            model_uri = f"models:/{MLFLOW_EXPERIMENT_NAME}/{model_version}"
            pipeline = mlflow.sklearn.load_model(model_uri=model_uri)

            # Load processor artifact
            local_processor_path = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path=MLFLOW_PROCESSOR_PATH
            )
            processor = joblib.load(local_processor_path)

            return pipeline, processor

        except Exception as e:
            raise RuntimeError(
                f"Failed to load artifacts for version '{model_version}': {e}"
            ) from e

    def list_all_aliases(self) -> Dict[str, str]:
        """List all current aliases and their model versions."""
        try:
            # Get all model versions
            all_versions = self.client.search_model_versions(
                f"name='{MLFLOW_EXPERIMENT_NAME}'"
            )
            aliases_map = {}

            for version in all_versions:
                if hasattr(version, "aliases") and version.aliases:
                    for alias in version.aliases:
                        aliases_map[alias] = version.version

            return aliases_map

        except Exception as e:
            self.logger.error(f"Failed to list aliases: {e}")
            # Return empty dict because no aliases existing is a valid state
            return {}

    def migrate_stages_to_aliases(self) -> None:
        """Migrate existing stage-based models to aliases."""
        try:
            print("Migrating stages to aliases...")

            # Map of stages to migrate
            stage_migrations = {
                "Production": "production",
                "Staging": "staging",
                "Archived": "archived",
            }

            for stage, alias in stage_migrations.items():
                try:
                    # This will show deprecation warning but still works
                    versions = self.client.get_latest_versions(
                        name=MLFLOW_EXPERIMENT_NAME, stages=[stage]
                    )

                    if versions:
                        version = versions[0].version
                        self.set_model_alias(version, alias)
                        print(
                            f"✅ Migrated stage '{stage}' to alias '{alias}' for version {version}"
                        )

                except Exception as e:
                    print(f"⚠️  Could not migrate {stage}: {e}")

            print("✅ Migration complete!")

        except Exception as e:
            print(f"❌ Migration failed: {e}")
