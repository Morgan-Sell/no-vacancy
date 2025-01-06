import os
from pathlib import Path
from typing import Any

import joblib

from app.config import PIPELINE_DIR, PIPELINE_SAVE_FILE, get_logger
from app.services.config_services import DATA_PATHS
from app.services.pipeline import NoVacancyPipeline


def handle_error_dm(logger, error_type, message, exception):
    logger.error(f"{message}: {exception}")
    raise error_type(f"{message}: {exception}")


# TODO: Change log savings from local directory to cloud provider storage, e.g., AWS S3.
class DataManagement:
    """
    Handles saving, loading, and managing the pipeline.
    """

    def __init__(self):
        self.logger = get_logger(logger_name=__name__)
        self.pipeline_path = Path(DATA_PATHS["model_save_path"])

    def save_pipeline(self, pipeline) -> None:
        try:
            if not isinstance(pipeline, NoVacancyPipeline):
                raise TypeError(
                    "The pipeline to be saved must be an instance of NoVacancyPipeline."
                )

            joblib.dump(pipeline, self.pipeline_path)
            self.logger.info(f"✅ Pipeline successfully saved at {self.pipeline_path}")

        except Exception as e:
            handle_error_dm(self.logger, type(e), "❌ Error during pipeline saving", e)


    def load_pipeline(self) -> NoVacancyPipeline:
        try:
            if not self.pipeline_path.exists():
                raise FileNotFoundError(
                    f"Pipeline file not found at {self.pipeline_path}"
                )

            pipeline = joblib.load(self.pipeline_path)

            if not isinstance(pipeline, NoVacancyPipeline):
                raise TypeError(
                    "Loaded pipeline is not an instance of NoVacancyPipeline."
                )

            self.logger.info(
                f"✅ Pipeline successfully loaded from {self.pipeline_path}"
            )
            return pipeline

        except Exception as e:
            handle_error_dm(self.logger, type(e), "❌ Error during pipeline loading", e)

    def delete_pipeline(self) -> None:
        try:
            if not self.pipeline_path.exists():
                raise FileNotFoundError(
                    f"Pipeline file not found at {self.pipeline_path}"
                )

            # Delete file as specified location
            self.pipeline_path.unlink()
            self.logger.info(
                f"✅ Pipeline successfully deleted from {self.pipeline_path}"
            )

        except Exception as e:
            handle_error_dm(self.logger, type(e), "❌ Error during pipeline deletion", e)
