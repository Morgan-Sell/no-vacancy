from services.mlflow_utils import MLflowArtifactLoader
from services.validation.base import ModelValidator


class ManualValidator(ModelValidator):
    """
    Manual validation strategy for data scientist approval.
    Checks MLflow tags and model stage for manual validation markers.
    """

    def validate(self, model_version: str) -> bool:
        """
        Check if model has been manually validated by data scientist.

        Args:
            model_version: The model version to validate.

        Returns:
            bool: True if manually validated and in Staging, otherwise False.

        """
        try:
            loader = MLflowArtifactLoader()
            return loader.check_manual_validation_status(model_version)
        except Exception:
            # Fail safe if validation status can't be checked.
            return False
