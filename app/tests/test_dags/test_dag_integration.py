"""
Lightweight DAG structure tests that avoid SQLAlchemy compatibility issues.
Tests DAG structure and basic functionality without importing full application stack.
"""

import pytest
from airflow.models import DagBag


class TestDAGStructure:
    """Test DAG structure without heavy application dependencies."""

    @pytest.fixture(scope="class")
    def dag_bag(self):
        """Load the DAG for testing"""
        return DagBag(dag_folder="/opt/airflow/dags", include_examples=False)

    def test_dag_loads_successfully(self, dag_bag):
        """Test that the DAG loads without import errors"""
        assert (
            len(dag_bag.import_errors) == 0
        ), f"DAG import errors: {dag_bag.import_errors}"

    def test_training_pipeline_exists(self, dag_bag):
        """Test that the training pipeline DAG exists"""
        dag = dag_bag.get_dag("training_pipeline")
        assert dag is not None, "training_pipeline DAG not found"

    def test_training_pipeline_structure(self, dag_bag):
        """Test that all expected tasks exist"""
        dag = dag_bag.get_dag("training_pipeline")

        expected_tasks = [
            "import_csv_data",
            "train_and_register_model",
            "generate_predictions",
            "validate_model_artifacts",
            "notify_completion",
        ]

        actual_tasks = [task.task_id for task in dag.tasks]
        for expected_task in expected_tasks:
            assert expected_task in actual_tasks, f"Missing task: {expected_task}"

    def test_task_dependencies(self, dag_bag):
        """Test that tasks have correct dependencies"""
        dag = dag_bag.get_dag("training_pipeline")

        # Get tasks
        import_task = dag.get_task("import_csv_data")
        train_task = dag.get_task("train_and_register_model")
        predict_task = dag.get_task("generate_predictions")
        validate_task = dag.get_task("validate_model_artifacts")
        notify_task = dag.get_task("notify_completion")

        # Test linear dependencies
        assert train_task in import_task.downstream_list
        assert predict_task in train_task.downstream_list
        assert validate_task in predict_task.downstream_list
        assert notify_task in validate_task.downstream_list

    def test_dag_configuration(self, dag_bag):
        """Test DAG scheduling and configuration"""
        dag = dag_bag.get_dag("training_pipeline")

        assert dag.schedule_interval == "@weekly"
        assert dag.catchup is False
        assert dag.max_active_runs == 1
