from abc import ABC, abstractmethod
from typing import Any, Dict


class ModelValidator(ABC):
    """
    Abstract base class for all model validators.
    """

    @abstractmethod
    def validate(self, model_version: str) -> Dict[str, Any]:
        """Validate a model version and return results"""
        pass
