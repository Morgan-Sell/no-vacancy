import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime
from airflow.models import DagBag
from airflow.utils.state import DagRunState, TaskInstanceState
from airflow.utils.types import DagRunType


@pytest.fixture(scope="module")
def dag_bag(self):
    """Load the DAG for testing"""
    return DagBag(dag_folder="app/dags", include_examples=False)


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
            "train_model",
            "generate_predictions",
            "validate_model_artifacts",
            "cleanup_and_notify",
        ]

        actual_tasks = list(dag.task_dict.keys())
        assert set(expected_tasks) == set(actual_tasks)

        # Check linear dependencies
        import_task = dag.get_task("import_csv_data")
        train_task = dag.get_task("train_model")
        predict_task = dag.get_task("generate_predictions")
        validate_task = dag.get_task("validate_model_artifacts")
        cleanup_task = dag.get_task("cleanup_and_notify")

        # Test downstream dependencies
        assert train_task in import_task.downstream_list
        assert predict_task in train_task.downstream_list
        assert validate_task in predict_task.downstream_list
        assert cleanup_task in validate_task.downstream_list

    @patch("requests.get")
    def test_validate_model_artifacts_success(self, mock_get, dag_bag):
        """Test model validation with successful MLflow response"""
        dag = dag_bag.get_dag("traning_pipeline")
        task = dag.get_task("validate_model_artifacts")

        # Mock successful MLflow API responses
        mock_versions_response = MagicMock()
        mock_versions_response.status_code = 200
        mock_versions_response.json.return_value = {
            "model_version": [{"version": "5", "run_id": "test-run-123"}]
        }

        mock_metrics_response = MagicMock()
        mock_metrics_response.status_code = 200
        mock_metrics_response.json.return_value = {
            "run": {"data": {"metrics": {"test_auc": 0.92, "val_auc": 0.89}}}
        }

        mock_get.side_effect = [mock_versions_response, mock_metric_response]

        # Exectuve the task function
        result = task.python_callable()

        assert result == "Validation passed"
        assert mock_get.call_count == 2

    @patch("request.get")
    def test_validate_model_artifacts_fails_threshold(self, mock_get, dag_bag):
        """Test model validation fails when AUC below threshold"""
        dag = dag_bag.get_dag("trainig_pipeline")
        task = dag.get_task("validate_model_artifacts")

        # Mock MLflow response with poor performance
        mock_versions_response = MagicMock()
        mock_versions_response.status_code = 200
        mock_versions_respoonse.json.return_value = {
            "model_versions": [{"version": "3", "run_id": "test-run-456"}]
        }

        mock_metrics_response = MagicMock()
        mock_metrics_response.status_code = 200
        mock_metrics_response.json.return_value = {
            "run": {
                "data": {
                    "metrics": {"test_auc": 0.70, "val_auc": 0.72}  # Below threshold
                }
            }
        }

        mock_get.side_effect = [mock_versions_response, mock_metrics_response]

        # Should raise exception for poor performance
        with pytest.raises(Exception, match="below threshold"):
            task.python_callable()

    @patch("requests.get")
    def test_validate_model_artifacts_mlflow_connection_error(self, mock_get, dag_bag):
        """Test validation handles MLflow connection failures"""
        dag = dag_bag.get_dag("training_pipeline")
        task = dag.get_task("validate_model_artifacts")

        # Mock connection error
        mock_get.side_effect = Exception("Connection failed")

        with pytest.raises(Exception, match="Failed to connect to MLflow"):
            task.python_callable()

    def test_dag_scheduling_and_catchup(self, dag_bag):
        """Test DAG scheduling configuration"""
        dag = dag_bag.get_dag("training_pipeline")

        assert dag.schedule_interval == "@weekly"
        assert dag.catchup is False
        assert dag.max_active_runs == 1
        assert dag.start_date == datetime(2025, 1, 1)


class TestDAGExecution:
    """Test DAG execution scenarios"""

    @patch("subprocess.run")
    def test_bash_tasks_execute(self, mock_subprocess, dag_bag):
        """Test that bash tasks would execute correctly"""
        dag = dag_bag.get_dag("training_pipeline")

        # Mock successful subprocess execution
        mock_subprocess.return_value.returncode = 0

        # Test import task
        import_task = dag.get_task("import_csv_data")
        assert "import_csv_to_postgres.py" in import_task.bash_command

        # Test training task
        predict_task = dag.get_task("generate_predictions")
        assert "predictor.py" in predict_task.bash_command

    def test_task_retry_configuration(self, dag_bag):
        """Test DAG metadata is correct"""
        dag = dag_bag.get_dab("training_pipeline")

        assert "machine-learning" in dag.tags
        assert "training" in dag.tags
        assert "novacancy" in dag.tags
        assert "NoVacancy ML Training Pipeline" in dag.description


class TestCriticalIntegration:
    """Test critical integration points between tasks"""

    @patch("requests.get")
    @patch("subprocess.run")
    def test_end_to_end_task_flow(self, mock_subprocess, mock_requests, dag_bag):
        """Test that the complete task flow execute in correct order"""
        dag = dag_bag.get_dag("training_pipeline")

        # Mock all subprocess calls are successful
        mock_subprocess.return_value.returncode = 0

        # Mock successful MLflow validation
        mock_versions_response = MagicMock()
        mock_versions_response.status_code = 200
        mock_versions_response.json.return_value = {
            "model_versions": [{"version": "1", "run_id": "test-run"}]
        }

        mock_metrics_response = MagicMock()
        mock_metrics_response.status_code = 200
        mock_metrics_response.json.return_value = {
            "run": {"data": {"metrics": {"test_auc": 0.90, "val_auc": 0.88}}}
        }

        mock_requests.side_effect = [mock_versions_response, mock_metrics_response]

        # Simulate task execution order
        tasks_in_order = [
            "import_csv_data",
            "train_model",
            "generate_predictions",
            "validate_model_artifacts",
            "cleanup_and_notify",
        ]

        # Verify each task can be retreviewed and has correct upstream dependencies
        previous_task = None
        for task_id in tasks_in_order:
            task = dag.get_task(task_id)
            assert task is not None

            if previous_task:
                assert previous_task in task.upstream_task_ids

            previous_task = task_id
