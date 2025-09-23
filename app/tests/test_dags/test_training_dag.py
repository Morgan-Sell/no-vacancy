import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime
from airflow.models import DagBag
from airflow.utils.state import DagRunState, TaskInstanceState
from airflow.utils.types import DagRunType


@pytest.fixture(scope="module")
def dag_bag():
    """Load the DAG for testing"""
    return DagBag(dag_folder="/opt/airflow/dags", include_examples=False)


class TestTrainingPiplelineDAG:
    """Test critical activities of the trianing pipeline DAG"""

    def test_dag_loads_successfully(self, dag_bag):
        """Test that the DAG loads without errors"""
        dag = dag_bag.get_dag("training_pipeline")
        assert dag is not None
        assert len(dag_bag.import_errors) == 0

    def test_dag_structure(self, dag_bag):
        """Test DAG has correct tasks and dependencies"""
        dag = dag_bag.get_dag("training_pipeline")

        # Check all expected tasks exists
        expected_tasks = [
            "import_csv_data",
            "train_and_register_model",
            "generate_predictions",
            "validate_model_artifacts",
            "cleanup_and_notify",
        ]

        actual_tasks = list(dag.task_dict.keys())
        assert set(expected_tasks) == set(actual_tasks)

        # Check linear dependencies
        import_task = dag.get_task("import_csv_data")
        train_task = dag.get_task("train_and_register_model")
        predict_task = dag.get_task("generate_predictions")
        validate_task = dag.get_task("validate_model_artifacts")
        cleanup_task = dag.get_task("cleanup_and_notify")

        # Test downstream dependencies
        assert train_task in import_task.downstream_list
        assert predict_task in train_task.downstream_list
        assert validate_task in predict_task.downstream_list
        assert cleanup_task in validate_task.downstream_list

    def test_dag_scheduling_and_catchup(self, dag_bag):
        """Test DAG scheduling configuration"""
        dag = dag_bag.get_dag("training_pipeline")

        assert dag.schedule_interval == "@weekly"
        assert dag.catchup is False
        assert dag.max_active_runs == 1
        assert dag.start_date.date() == datetime(2025, 1, 1).date()
