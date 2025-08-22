import logging
import os
import sys
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from logging.handlers import TimedRotatingFileHandler
from os.path import abspath, dirname, join

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# -- Logging Config --
PACKAGE_ROOT = abspath(dirname(__file__))

FORMATTER = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s -" "%(funcName)s:%(lineno)d - %(message)s"
)
LOG_DIR = join(PACKAGE_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = join(LOG_DIR, "no_vacancy_api.log")

# -- Data Management --
PIPELINE_DIR = "app/models"
PIPELINE_SAVE_FILE = "no_vacancy_pipeline.pkl"

# Versioning
__api_version__ = "0.0.0"
__model_version__ = "0.0.3"


# Create a custom logger
def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler


def get_file_handler():
    file_handler = TimedRotatingFileHandler(LOG_FILE, when="midnight")
    file_handler.setFormatter(FORMATTER)
    file_handler.setLevel(logging.WARNING)
    return file_handler


def get_logger(*, logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.addHandler(get_console_handler())
    logger.addHandler(get_file_handler())
    logger.propagate = False
    return logger


# -- General Confg --
# Determin environment (test, prod)
ENV = os.getenv("ENV", "PROD").upper()

# -- Postgres Config (Production) --
# General DB config

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_CONNECT_TIMEOUT = 120  # in seconds
DB_PORT = os.getenv("DB_PORT")

# Bronze DB config
BRONZE_DB_HOST = os.getenv("BRONZE_DB_HOST")
BRONZE_DB = os.getenv("BRONZE_DB")
BRONZE_DB_PORT = os.getenv("BRONZE_DB_PORT")

# Silver DB config
SILVER_DB_HOST = os.getenv("SILVER_DB_HOST")
SILVER_DB = os.getenv("SILVER_DB")
SILVER_DB_PORT = os.getenv("SILVER_DB_PORT")


# Gold DB config
GOLD_DB_HOST = os.getenv("GOLD_DB_HOST")
GOLD_DB = os.getenv("GOLD_DB")
GOLD_DB_PORT = os.getenv("GOLD_DB_PORT")

# Test DB config
TEST_DB_USER = os.getenv("TEST_DB_USER")
TEST_DB_PASSWORD = os.getenv("TEST_DB_PASSWORD")
TEST_DB_HOST = os.getenv("TEST_DB_HOST")
TEST_DB = os.getenv("TEST_DB")
TEST_DB_PORT = os.getenv("TEST_DB_PORT")

# MLflow config
MLFLOW_DB_HOST = os.getenv("MLFLOW_DB_HOST")
MLFLOW_DB = os.getenv("MLFLOW_DB")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")


# CSV File Paths
DATA_DIR = "./data"
RAW_DATA_FILE_PATH = os.path.join(DATA_DIR, "bookings_raw.csv")


# Storage
CSV_HASH_TABLE = "csv_hashes"
RAW_DATA_TABLE = "raw_data"

# -- Docker Config --

# Timeout configurations (in seconds)
TRAINING_DEPLOYMENT_TIMEOUT = 3600  # 1 hour
INFERENCE_DEPLOYMENT_TIMEOUT = 60  # 1 minute
MLFLOW_DEPLOYMENT_TIMEOUT = 30  # 30 seconds

# Container names
TRAINING_CONTAINER = "training-container"
INFERENCE_CONTAINER = "inference-container"
MLFLOW_CONTAINER = "mlflow"

# Docker compose commands
DOCKER_COMPOSE_RESTART_CMD = ["docker", "compose", "restart"]
DOCKER_COMPOSE_RUN_CMD = ["docker", "compose", "run", "--rm"]
DOCKER_COMPOSE_TRAINING_CMD = [
    "docker",
    "compose",
    "--profile",
    "training",
    "run",
    "--rm",
]


# -- Continuous Deployment --


class DeploymentMode(Enum):
    """Deployment modes for different container strategies."""

    INFERENCE_CONTAINER_RESTART = "inference_container_restart"
    TRAINING_CONTAINER_RUN = "training_container_run"
    MLFLOW_ONLY = "mlflow_only"
    KUBERNETES = "kubernetes"  # Placeholder for future addtions


@dataclass
class CDConfig:
    """
    Configuration for Continuous Deployment (CD) settings.
    """

    target_environment: str
    require_manual_validation: bool
    deployment_mode: DeploymentMode
    inference_container_name: str = "inference-container"
    training_container_name: str = "training-container"
    mlflow_container_name: str = "mlflow"

    @classmethod
    def for_production_inference(cls):
        """Production configuration with container restart."""
        return cls(
            target_environment="production",
            require_manual_validation=True,
            deployment_mode=DeploymentMode.INFERENCE_CONTAINER_RESTART,
            inference_container_name="inference-container",
        )

    @classmethod
    def for_automated_training(cls):
        """Configuration for automated training workflows."""
        return cls(
            target_environment="training",
            require_manual_validation=False,
            deployment_mode=DeploymentMode.TRAINING_CONTAINER_RUN,
            training_container_name="training-container",
        )

    @classmethod
    def for_staging_mlflow(cls):
        """Staging configuration - MLflow only."""
        return cls(
            target_environment="staging",
            require_manual_validation=False,
            deployment_mode=DeploymentMode.MLFLOW_ONLY,
        )

    @classmethod
    def for_development_training(cls):
        """Development configuration for training experiments."""
        return cls(
            target_environment="development",
            require_manual_validation=False,
            deployment_mode=DeploymentMode.TRAINING_CONTAINER_RUN,
            training_container_name="training-container",
        )


# -- Orchestration --
DAG_DEFAULT_ARGS = {
    "owner": "mickey-mouse",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
    "email_on_retry": False,
}
