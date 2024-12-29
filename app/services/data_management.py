import os

import joblib
from pathlib import Path
from typing import Any
from app.config import PIPELINE_SAVE_FILE, PIPELINE_DIR, get_logger


# TODO: Change log savings from local directory to cloud provider storage, e.g., AWS S3.
class DataManagement:
    """
    Handles saving, loading, and managing the pipeline.
    """
    def __init__(self):
        self.logger = get_logger(logger_name=__name__)
        self.pipeline_path = Path(PIPELINE_DIR) / PIPELINE_SAVE_FILE
  
    def save_pipeline(self, pipeline_to_persist: Any) -> None:
        """
        Save the pipeline locally to the specified PIPELINE_DIR.
        """
        try:
            self.pipeline_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(pipeline_to_persist, self.pipeline_path)
            self.logger.info(f"Pipeline successfully saved at {self.pipeline_path}")
        except Exception as e:
            self.logger.error(f"Failed to save pipeline: {e}")
            raise RuntimeError(f"Failed to save pipeline: {e}")
   
    def load_pipeline(self) -> Any:
        if not self.pipeline_path.exists():
            error_msg = f"❌ No pipeline found at {self.pipeline_path}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        try:
            pipeline = joblib.load(self.pipeline_path)
            self.logger.info(f"✅ Pipeline successfuly loaded from {self.pipeline_path}")
            return pipeline
        except Exception as e:
            self.logger.error(f"❌ Failed to load pipeline: {e}")
            raise RuntimeError(f"Failed to load pipeline: {e}"
  
    def delete_pipeline(self) -> None:
        try:
            if self.pipeline_path.is_file():
                os.remove(self.pipeline_path)
                self.logger.info(f"✅ Pipeline successfully deleted at {self.pipeline_path}")
            else:
                self.logger.warning(f"❌ No pipeline found to delete at {self.pipeline_path}")
        except Exception as e:
            self.logger.error(f"❌ Failed to delete pipeline: {e}")
            raise RuntimeError(f"Failed to delete pipeline: {e}")
