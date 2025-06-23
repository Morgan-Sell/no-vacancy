from dataclasses import dataclass
from typing import Dict, List


@dataclass
class CDConfig:
    """
    Configuration for Continuous Deployment (CD) settings.
    """

    validation_thresholds: Dict[str, float]
    deployment_environments: List[str]
    rollback_timeout: int
    health_check_retries: int
    monitoring_enable: bool = True

    @classmethod
    def from_env(cls, environment: str = "production"):
        """
        Create configuration for specific environment.
        classmethod allows for different settings of CDConfig based on the environment.
        """
        if environment == "staging":
            return cls(
                validation_thresholds={"min_auc": 0.80, "drift_threshold": 0.15},
                deployment_environments=["staging"],
                rollback_timeout=30,
                health_check_retries=2,
                monitoring_enable=True,
            )

        elif environment == "production":
            return cls(
                validation_thresholds={"min_auc": 0.85, "drift_threshold": 0.1},
                deployment_environments=["staging", "production"],
                rollback_timeout=60,
                health_check_retries=5,
                monitoring_enable=True,
            )

        elif environment == "development":
            return cls(
                validation_thresholds={"min_auc": 0.75, "drift_threshold": 0.2},
                deployment_environments=["development"],
                rollback_timeout=15,
                health_check_retries=1,
                monitoring_enable=False,
            )
        else:
            raise ValueError(
                f"Unknown environment: {environment}. Supported environments are 'staging', 'production', and 'development'."
            )
