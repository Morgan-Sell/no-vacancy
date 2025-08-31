import pytest


@pytest.fixture(autouse=True)
def mock_mlflow():
    """
    Override the global autouse of MLflow mock in tests/conftest.py.
    DAG parsing doens't require MLflow.
    """
    yield
