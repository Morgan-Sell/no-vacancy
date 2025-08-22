import asyncio
import sys
from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import DAG_DEFAULT_ARGS
from scripts.import_csv_to_postgres import main as import_data
from services import MLFLOW_AUC_THRESHOLD
from services.mlflow_utils import MLflowArtifactLoader
from services.trainer import train_pipeline

dag = DAG(
    "training_pipeline",
    default_args=DAG_DEFAULT_ARGS,
    description="ML Training Pipeline",
    schedule_interval="@weekly",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
)


def import_csv_data(**context):
    """Import raw data from CSV"""
    import_data()
    return "Data imported"


def train_model(**context):
    """Execute NoVacancyDataProcessing and NoVacancyPipeline"""
    asyncio.run(train_pipeline())
    return "Model trained"


def validate_artifacts(**context):
    """Validate training artifacts in MLflow"""
    loader = MLflowArtifactLoader()
    metadata = loader.get_artifact_metadata_by_alias("production")

    if not metadata or not metadata.get("version"):
        raise Exception("No model artifacts found")

    # Check performance
    metrics = metadata.get("metrics", {})
    test_auc = metrics.get("test_auc", 0)

    if test_auc < MLFLOW_AUC_THRESHOLD:
        raise Exception(
            f"Model AUC of {test_auc} is below {MLFLOW_AUC_THRESHOLD} threshold"
        )

    print(f"✅ Validation passed - Version: {metadata['version']}")
    return "Validation passed"


# Define tasks with single responsibilities
import_task = PythonOperator(
    task_id="import_data",
    python_callable=import_csv_data,
    dag=dag,
)

train_task = PythonOperator(task_id="train_model", python_callable=train_model, dag=dag)

validate_task = PythonOperator(
    task_id="validate_artifacts", python_callable=validate_artifacts, dag=dag
)

# Linear pipeline
import_task >> train_task >> validate_task
