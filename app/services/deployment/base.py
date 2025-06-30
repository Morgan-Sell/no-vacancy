from abc import ABC, abstractmethod
from typing import Optional


class DeploymentStrategy(ABC):
    """Abstract deployment strategy for container architectures."""

    @abstractmethod
    def deploy(self, model_version: Optional[str] = None) -> dict:
        """Deploy model. Returns deployment status."""
        raise NotImplementedError
