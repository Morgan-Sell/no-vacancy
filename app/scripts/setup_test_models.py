"""
Script to set up test model for CD pipeline testing.
This script creates a model in the Staging stage of MLflow to trigger the CD workflow.
"""

import sys
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from services import MLFLOW_EXPERIMENT_NAME
from services.mlflow_utils import MLflowArtifactLoader


def setup_test_models():
    """Create test models and move one to Staging for CD pipeline testing."""
    try:
        loader = MLflowArtifactLoader()

        # Check if any models already exist
        all_versions = loader.client.search_model_versions(
            f"name='{MLFLOW_EXPERIMENT_NAME}'"
        )

        if not all_versions:
            print("❌ No models found. Please run 'python services/trainer.py' first.")
            return False

        # Get the latest model (should be in Production from trainer.py)
        latest_version = max(all_versions, key=lambda x: int(x.version))
        print(f"Found model version: {latest_version.version}")

        # Check if staging alias already exists
        staging_model = loader.get_model_by_alias("staging")
        if staging_model:
            print(f"Model version {staging_model.version} already has staging alias.")
            print("✅ CD pipeline is ready to test!")
            return True

        # If it's in Production, create a copy by transitioning to Staging
        if latest_version.current_stage == "Production":
            print(
                f"Moving model version {latest_version.version} to Staging for testing..."
            )

            # Transition to Staging (this creates a staging model)
            loader.client.transition_model_version_stage(
                name=MLFLOW_EXPERIMENT_NAME,
                version=latest_version.version,
                stage="Staging",
                archive_existing_versions=False,
            )

            print(f"Model version {latest_version.version} is now in Staging")
            print("✅ CD pipeline can now be tested!")
            return True

        elif latest_version.current_stage == "Staging":
            print(f"Model version {latest_version.version} is already in Staging")
            print("✅ CD pipeline is ready to test!")
            return True

        else:
            print(
                f"Model is in '{latest_version.current_stage}' stage. Moving to Staging..."
            )
            loader.client.transition_model_version_stage(
                name=MLFLOW_EXPERIMENT_NAME,
                version=latest_version.version,
                stage="Staging",
            )
            print(f"✅ Model version {latest_version.version} moved to Staging")
            return True

    except Exception as e:
        print(f"❌ Error setting up test models: {e}")
        return False


if __name__ == "__main__":
    if setup_test_models():
        sys.exit(0)
    else:
        sys.exit(1)
