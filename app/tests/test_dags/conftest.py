import pytest
import importlib.util

# If Airflow isn't installed (the unit job), tell pytest to ignore all tests in test_dags
if importlib.util.find_spec("airflow") is None:
    collect_ignore_glob = ["*.py"]  # don't attempt to import anything here


@pytest.fixture(autouse=True)
def mock_mlflow():
    """
    Override the global autouse of MLflow mock in tests/conftest.py.
    DAG parsing doens't require MLflow.
    """
    yield
