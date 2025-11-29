"""
Configuration management for NoVacancy frontend.

Uses environment variables with sensible defaults for local development.
"""

import os


class Config:
    """Application configuration from environment variables."""

    # Flask settings
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-prod")
    DEBUG = os.getenv("FLASK_DEBUG", "false").lower() == "false"

    # Backend service URLs
    FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")
    AIRFLOW_URL = os.getenv("AIRFLOW_URL", "http://localhost:8080")

    # Airflow credentials
    AIRFLOW_USERNAME = os.getenv("AIRFLOW_USERNAME", "homer")
    AIRFLOW_PASSWORD = os.getenv("AIRFLOW_PASSWORD", "waffles")

    # Airflow DAG configuration
    TRAINING_DAG_ID = "training_pipeline"

    # Request timeouts (seconds)
    PREDICT_TIMEOUT = 30
    AIRFLOW_TIMEOUT = 10
