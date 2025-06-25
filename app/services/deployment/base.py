from abc import ABC, abstractmethod


class DeploymentStrategy(ABC):
    """Abstract deployment strategy for container architectures."""

    @abstractmethod
    def deploy(self, model_version: str) -> dict:
        """Deploy model. Returns deployment status."""
        pass
