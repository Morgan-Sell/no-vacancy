from pathlib import Path
from typing import NoReturn, Tuple

import joblib
from config import get_logger
from services import DATA_PATHS
from services.pipeline import NoVacancyPipeline
from services.preprocessing import NoVacancyDataProcessing


def handle_error_dm(logger, error_type, message, exception) -> NoReturn:
    logger.error(f"{message}: {exception}")
    raise error_type(f"{message}: {exception}")


# TODO: Change log savings from local directory to cloud provider storage, e.g., AWS S3.
class PipelineManagement:
    """
    Handles saving, loading, and managing the pipeline.
    """

    def __init__(self, pipeline_path: str = DATA_PATHS["model_save_path"]) -> None:
        self.logger = get_logger(logger_name=__name__)
        self.pipeline_path = Path(pipeline_path)

    def save_pipeline(
        self, pipeline: NoVacancyPipeline, processor: NoVacancyDataProcessing
    ) -> None:
        self.__validate_pipeline_and_processor(pipeline, processor)

        try:
            # Ensure the directory exists before saving
            self.pipeline_path.parent.mkdir(parents=True, exist_ok=True)

            # Save both pipeline and processor as a dictionary
            joblib.dump(
                {"pipeline": pipeline, "processor": processor},
                self.pipeline_path,
            )

            self.logger.info(
                f"✅ Pipeline and processor successfully saved at {self.pipeline_path}"
            )

        except Exception as e:
            handle_error_dm(self.logger, type(e), "❌ Error during pipeline saving", e)

    def load_pipeline(self) -> Tuple[NoVacancyPipeline, NoVacancyDataProcessing]:
        try:
            # Generate a comprehensive error message if the pipeline path is not found
            if not self.pipeline_path.exists():
                raise FileNotFoundError(
                    f"Pipeline file not found at {self.pipeline_path}"
                )

            # Load as dictionary
            artifacts = joblib.load(self.pipeline_path)
            pipeline = artifacts.get("pipeline")
            processor = artifacts.get("processor")

            self.__validate_pipeline_and_processor(pipeline, processor)

            self.logger.info(
                f"✅ Pipeline and processor successfully loaded from {self.pipeline_path}"
            )
            return pipeline, processor

        except FileNotFoundError as e:
            handle_error_dm(self.logger, FileExistsError, "❌ Pipeline not found", e)

        except (ValueError, AttributeError, TypeError) as e:
            handle_error_dm(
                self.logger, type(e), "❌ Invalid pipeline or processor format", e
            )

        except Exception as e:
            handle_error_dm(
                self.logger,
                RuntimeError,
                "❌ Unexpected error during pipeline loading",
                e,
            )

    def delete_pipeline(self) -> None:
        try:
            if not self.pipeline_path.exists():
                raise FileNotFoundError(
                    f"Pipeline file not found at {self.pipeline_path}"
                )

            # Delete file as specified location
            self.pipeline_path.unlink()
            self.logger.info(
                f"✅ Pipeline and processor successfully deleted from {self.pipeline_path}"
            )

        except Exception as e:
            handle_error_dm(
                self.logger, type(e), "❌ Error during pipeline deletion", e
            )

    def __validate_pipeline_and_processor(
        self, pipeline: NoVacancyPipeline, processor: NoVacancyDataProcessing
    ) -> None:
        if not isinstance(pipeline, NoVacancyPipeline):
            raise TypeError(
                "❌ Error during pipeline validation: The pipeline must be an instance of NoVacancyPipeline"
            )

        if not isinstance(processor, NoVacancyDataProcessing):
            raise TypeError(
                "❌ Error during processor validation: The processor must be an instance of NoVacancyDataProcessing"
            )
