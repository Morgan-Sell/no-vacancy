"""
Training Orchestrator - Coordinates the complete training workflow
This is the entry point for the training container and handles:
1. Data validation and preparation
2. Model training execution
3. MLflow artifact management
4. Error handling and logging
"""

import asyncio
import sys
from pathlib import Path

# Add app directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_logger
from scripts.import_csv_to_postgres import main as import_data
from services.mlflow_utils import MLflowArtifactLoader
from services.trainer import train_pipeline

from app.services import MLFLOW_EXPERIMENT_NAME

logger = get_logger(logger_name=__name__)


class TrainingOrchestrator:
    """
    Orchestrates the complete training workflow for the training container.
    Follows the Single Responsibility principle by coordinating different components.
    """

    def __init__(self):
        self.logger = logger

    async def execute_training_workflow(self) -> bool:
        """
        Execute the complete training workflow.
        Returns: True if successful, False if failed.
        """

        try:
            self.logger.info("Starting training orchestration workflow")

            # Step 1: Import raw data from csv to Bronze DB
            self.logger.info("Step 1: Importing raw data to Bronze database")
            try:
                import_data()
                self.logger.info("✅ Data import completed successfully")
            except Exception as e:
                self.logger.error(f"❌ Data import failed: {e}")
                return False

            # Step 2: Execute training pipeline
            try:
                await train_pipeline()
                self.logger.info("✅ Model training completed successfully")
            except Exception as e:
                self.logger.error(f"❌ Model training failed: {e}")
                return False

            # Step 3: Validate training results
            if await self._validate_training_results():
                self.logger.info("✅ Training validation passed")
                self.logger.info(
                    "🎉 Training orchestration workflow completed successfully"
                )
                return True
            else:
                self.logger.error("❌ Training validation failed")
                return False

        except Exception as e:
            self.logger.error(
                f"❌ Training orchestration workflow failed with unexpected error: {e}"
            )
            return False

    async def _validate_training_results(self) -> bool:
        """
        Validate that training produced expected artifacts.
        This checks that the model was successfully trained and registered in MLflow.
        """
        try:
            loader = MLflowArtifactLoader()

            # Check for any registered model versions (not production alias)
            try:
                all_aliases = loader.list_all_aliases()

                # If not alias exist yet, check if model was registered at all
                if not all_aliases:
                    # Check if the model exists in registry
                    versions = loader.client.search_model_versions(
                        f"name='{MLFLOW_EXPERIMENT_NAME}'"
                    )
                    if versions:
                        latest = max(versions, key=lambda v: int(v.version))
                        self.logger.info(
                            f"✅ Model version {latest.version} registered successfully"
                        )
                        return True
                    else:
                        self.logger.error("❌ No model versions found in registry")
                        return False

                self.logger.info(f"✅ Model aliases found: {all_aliases}")
                return True

            except Exception as e:
                self.logger.error(f"❌ Could not validate MLflow artifacts: {e}")
                return False

        except Exception as e:
            self.logger.error(f"❌ Training result validation failed: {e}")
            return False


async def main():
    """
    Main entry point for the training container.
    This function is called by Dockerfile.training CMD.
    """

    orchestrator = TrainingOrchestrator()

    success = await orchestrator.execute_training_workflow()

    if success:
        logger.info("🎉 Training orchestration completed successfully")
        sys.exit(0)
    else:
        logger.error("❌ Training orchestration failed")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
