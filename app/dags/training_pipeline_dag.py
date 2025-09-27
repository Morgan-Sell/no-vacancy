"""
NoVacancy ML Training Pipeline - Airflow DAG

Uses PythonOperator with wrapper functions instead of DockerOperator for:
- Better error handling and logging integration with Airflow
- Tasks can share data with each other (model versions, metrics)
- Access to Airflow context (run dates, trigger types)
- Shared memory space without subprocess overhead
- Production-standard MLOps patterns

Trade-off: More complex than direct script execution, but enables
advanced workflow features and better observability.
"""

import sys
from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

sys.path.insert(0, "/opt/airflow/project/app")
from config import DAG_DEFAULT_ARGS
from services import MLFLOW_AUC_THRESHOLD, MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI

dag = DAG(
    "training_pipeline",
    default_args=DAG_DEFAULT_ARGS,
    description="NoVacancy ML Training Pipeline",
    schedule_interval="@weekly",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["machine-learning", "training", "novacancy"],
)


def import_csv_data(**context):
    """Import CSV data to Bronze database"""
    import asyncio

    from scripts.import_csv_to_postgres import main as import_main

    asyncio.run(import_main())


def train_and_register_model(**context):
    """Train model, process data, and register in MLflow"""
    import asyncio

    from services.trainer import train_pipeline

    asyncio.run(train_pipeline())


def generate_predictions(**context):
    """Generate predictions on test data and save to Gold DB"""
    import asyncio

    from db.db_init import silver_db
    from schemas.silver import TestData
    from services.predictor import load_test_data, make_prediction

    async def run_predictions():
        # Load test data from Silver DB
        data = await load_test_data(silver_db.create_session()(), TestData)

        # Generate predictions and save to Gold DB
        await make_prediction(data, already_processed=True)

    asyncio.run(run_predictions())


def validate_model_artifacts(**context):
    """Lightweight validation using HTTP request to MLflow API"""

    import requests

    try:
        # Get the latest production model via MLflow REST API
        response = requests.get(
            f"{MLFLOW_TRACKING_URI}/api/2.0/mlflow/model-versions/search",
            params={
                "filter": f"name={MLFLOW_EXPERIMENT_NAME}",
                "max_results": 1,
                "order_by": ["version_number DESC"],
            },
        )

        if response.status_code != 200:  # noqa: PLR2004
            raise Exception(f"MLflow API error: {response.status_code}")

        data = response.json()
        model_versions = data.get("model_versions", [])

        data = response.json()
        model_versions = data.get("model_versions", [])

        if not model_versions:
            raise Exception("No model versions found in MLflow")

        latest_version = model_versions[0]
        run_id = latest_version["run_id"]

        # Get run metrics
        metrics_response = requests.get(
            f"{MLFLOW_TRACKING_URI}/api/2.0/mlflow/runs/get", params={"run_id": run_id}
        )

        if metrics_response.status_code != 200:  # noqa: PLR2004
            raise Exception(f"Failed to get run metrics: {metrics_response.status}")

        run_data = metrics_response.json()
        metrics = run_data["run"]["data"]["metrics"]

        test_auc = float(metrics.get("test_auc", 0))
        val_auc = float(metrics.get("val_auc", 0))

        print(f"Model Version: {latest_version['version']}")
        print(f"Test AUC: {test_auc}")
        print(f"Validation AUC: {val_auc}")
        print(f"Required threshold: {MLFLOW_AUC_THRESHOLD}")

        if test_auc < MLFLOW_AUC_THRESHOLD:
            raise Exception(
                f"Model Test AUC ({test_auc}) below threshold ({MLFLOW_AUC_THRESHOLD})"
            )

        print(f"✅ Validation passed - Version: {latest_version['version']}")
        return "Validation passed"

    except requests.RequestException as e:
        raise Exception(f"Failed to connect to MLflow: {e}") from e


# Task #1: Import CSV data to Bronze database
import_data_task = PythonOperator(
    task_id="import_csv_data", python_callable=import_csv_data, dag=dag
)

# Task #2: Traing the model (data processing + model training + MLflow saving)
training_task = PythonOperator(
    task_id="train_and_register_model",
    python_callable=train_and_register_model,
    dag=dag,
)

# Task 3: Generate predictions on test data and save to Gold DB
predict_task = PythonOperator(
    task_id="generate_predictions",
    python_callable=generate_predictions,
    dag=dag,
)

# Task 4: Lightweight validation using MLflow REST API
validation_task = PythonOperator(
    task_id="validate_model_artifacts",
    python_callable=validate_model_artifacts,
    dag=dag,
)

# Task 5: Notify team that training has successfully completed
notify_task = BashOperator(
    task_id="notify_completion",
    bash_command="""
    echo "🎉 Training pipeline completed successfully!"
    echo "📊 Model artifacts saved to MLflow"
    echo "🔮 Validation saved to Gold database"
    echo "🔬 Ready for Data Scientist validation"
    """,
    dag=dag,
)

# Defin task dependencies - Linear pipeline as requested
import_data_task >> training_task >> predict_task >> validation_task >> notify_task
