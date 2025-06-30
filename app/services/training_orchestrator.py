"""
Training Orchestrator - Coordinates the complete training workflow
This is the entry point for the training container and handles:
1. Data validation and preparation
2. Model training execution
3. MLflow artifact management
4. Error handling and logging
"""

import sys
from pathlib import Path

# Add app directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_logger
from scripts.import_csv_to_postgres import main as import_data
from services.trainer import train_pipeline

logger = get_logger(logger_name=__name__)


class TrainingOrchestrator:
    """
    Orchestrates the complete training workflow for the training container.
    Follows the Single Responsibility principle by coordinating different components.
    """

    def __init__(self):
        self.logger = logger

    async def execute_training_workflow(self):
        """
        Execute the complete training workflow.
        Returns: True if successful, False if failed.
        """

        try:
            self.logger.info("Starting training orchestration workflow")

            # Step 1: Imporate raw data from csv to Bronze DB
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
            # if await self._validate_training_results():
            #     self.logger.info("✅ Training validation passed")
            #     self.logger.info(
            #         "🎉 Training orchestration workflow completed successfully"
            #     )
            #     return True
            # else:
            #     self.logger.error("❌ Training validation failed")
            #     return False

        except Exception as e:
            self.logger.error(
                f"❌ Training orchestration workflow failed with unexpected error: {e}"
            )
            return False
