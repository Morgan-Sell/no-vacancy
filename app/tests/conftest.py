import glob
import os
import shutil
import tempfile
import time
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import psycopg2
import pytest
from services.mlflow_utils import MLflowArtifactLoader
from config import TEST_DB, TEST_DB_HOST, TEST_DB_PASSWORD, TEST_DB_PORT, TEST_DB_USER
from feature_engine.encoding import OneHotEncoder
from feature_engine.imputation import CategoricalImputer
from services import (
    BOOKING_MAP,
    DATA_PATHS,
    IMPUTATION_METHOD,
    MONTH_ABBREVIATION_MAP,
    PRIMARY_KEY,
    VARIABLE_RENAME_MAP,
    VARS_TO_DROP,
    VARS_TO_IMPUTE,
    VARS_TO_OHE,
)
from services.pipeline import NoVacancyPipeline
from services.preprocessing import NoVacancyDataProcessing
from sklearn.ensemble import RandomForestClassifier
from tests import TEST_TABLE

# -- Helper Functions --


@pytest.fixture(scope="session")
def booking_data():
    data = {
        "Booking_ID": [f"INN0000{i}" for i in range(1, 21)],
        "number_of_adults": [
            1,
            1,
            2,
            1,
            1,
            2,
            2,
            1,
            2,
            1,
            2,
            1,
            2,
            2,
            1,
            1,
            2,
            1,
            1,
            2,
        ],
        "number_of_children": [
            1,
            0,
            1,
            0,
            0,
            2,
            1,
            1,
            0,
            2,
            1,
            0,
            1,
            2,
            0,
            1,
            1,
            0,
            2,
            1,
        ],
        "number_of_weekend_nights": [
            2,
            1,
            1,
            0,
            1,
            2,
            0,
            1,
            1,
            0,
            2,
            1,
            0,
            1,
            1,
            2,
            1,
            0,
            1,
            1,
        ],
        "number_of_week_nights": [
            5,
            3,
            3,
            2,
            2,
            4,
            3,
            5,
            2,
            1,
            6,
            4,
            3,
            2,
            1,
            5,
            6,
            2,
            3,
            4,
        ],
        "type of meal": [
            "Meal Plan 1",
            "Not Selected",
            "Meal Plan 1",
            "Meal Plan 1",
            "Not Selected",
            "Meal Plan 2",
            "Meal Plan 3",
            "Meal Plan 1",
            "Meal Plan 1",
            "Not Selected",
            "Meal Plan 3",
            "Meal Plan 2",
            "Meal Plan 1",
            "Not Selected",
            "Meal Plan 2",
            "Meal Plan 1",
            "Meal Plan 2",
            "Not Selected",
            "Meal Plan 3",
            "Meal Plan 1",
        ],
        "car parking space": [
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            0,
            0,
            1,
            1,
            0,
            0,
            1,
            1,
            0,
            1,
            0,
            1,
            1,
        ],
        "room type": [
            "Room_Type 1",
            "Room_Type 6",
            "Room_Type 3",
            "Room_Type 2",
            "Room_Type 4",
            "Room_Type 1",
            "Room_Type 2",
            "Room_Type 2",
            "Room_Type 1",
            "Room_Type 4",
            "Room_Type 4",
            "Room_Type 5",
            "Room_Type 1",
            "Room_Type 6",
            "Room_Type 7",
            "Room_Type 3",
            "Room_Type 1",
            "Room_Type 5",
            "Room_Type 6",
            "Room_Type 7",
        ],
        "lead_time": [
            224,
            5,
            1,
            211,
            48,
            150,
            35,
            60,
            20,
            10,
            30,
            45,
            60,
            15,
            25,
            55,
            70,
            33,
            77,
            85,
        ],
        "market_segment_type": [
            "Offline",
            "Online",
            "Airline",
            "Corporate",
            "Online",
            "Offline",
            "Online",
            "Corporate",
            "Online",
            "Online",
            "Airline",
            "Offline",
            "Corporate",
            "Online",
            "Airline",
            "Corporate",
            "Online",
            "Offline",
            "Corporate",
            "Airline",
        ],
        "repeated": [0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0],
        "P-C": [0, 0, 0, 3, 0, 0, 0, 0, 0, 1, 1, 0, 12, 0, 0, 4, 2, 0, 3, 1],
        "P-not-C": [5, 0, 24, 0, 0, 2, 1, 0, 0, 0, 0, 0, 48, 0, 0, 10, 15, 0, 5, 12],
        "average_price": [
            88.00,
            106.68,
            50.70,
            100.00,
            77.00,
            120.00,
            85.50,
            90.03,
            60.00,
            110.34,
            323.00,
            105.50,
            72.00,
            130.00,
            222.00,
            99.00,
            144.50,
            123.75,
            111.80,
            95.60,
        ],
        "special_requests": [
            0,
            1,
            0,
            1,
            0,
            0,
            1,
            1,
            0,
            1,
            1,
            0,
            1,
            0,
            1,
            1,
            0,
            1,
            0,
            1,
        ],
        "date_of_reservation": [
            "10/2/2015",
            "11/6/2018",
            "2/28/2018",
            "12/20/2017",
            "4/11/2018",
            "3/15/2019",
            "12/25/2020",
            "8/19/2016",
            "9/12/2020",
            "7/8/2021",
            "1/15/2022",
            "12/25/2020",
            "5/10/2019",
            "3/30/2021",
            "11/11/2022",
            "2/14/2021",
            "6/19/2022",
            "4/5/2023",
            "8/9/2023",
            "12/1/2023",
        ],
        "booking_status": [
            "Not_Canceled",
            "Not_Canceled",
            "Canceled",
            "Canceled",
            "Canceled",
            "Not_Canceled",
            "Not_Canceled",
            "Canceled",
            "Not_Canceled",
            "Canceled",
            "Not_Canceled",
            "Canceled",
            "Canceled",
            "Not_Canceled",
            "Not_Canceled",
            "Canceled",
            "Not_Canceled",
            "Canceled",
            "Not_Canceled",
            "Canceled",
        ],
    }
    df = pd.DataFrame(data)
    return df


@pytest.fixture(scope="function")
def preprocessed_booking_data(booking_data):
    """Returns preprocessed booking data ready for NoVacancyPipeline."""
    processor = NoVacancyDataProcessing(
        variable_rename=VARIABLE_RENAME_MAP,
        month_abbreviation=MONTH_ABBREVIATION_MAP,
        vars_to_drop=VARS_TO_DROP,
        booking_map=BOOKING_MAP,
    )
    df = booking_data.copy()
    X = df.drop(columns=["booking_status"])
    y = df["booking_status"]

    X_tr, y_tr = processor.fit_transform(X, y)
    return X_tr, y_tr


@pytest.fixture(scope="function")
def mock_read_csv(mocker, booking_data):
    """Mock pandas read_csv to return booking_data."""
    return mocker.patch("pandas.read_csv", return_value=booking_data)


@pytest.fixture(scope="function")
def sample_pipeline():
    """Provide a valid sample NoVacancyPipeline instance."""
    imputer = CategoricalImputer(
        imputation_method=IMPUTATION_METHOD, variables=VARS_TO_IMPUTE
    )
    encoder = OneHotEncoder(variables=VARS_TO_OHE)
    estimator = RandomForestClassifier()

    pipeline = NoVacancyPipeline(imputer, encoder, estimator, [PRIMARY_KEY])
    return pipeline


@pytest.fixture(scope="function")
def mock_pipeline(sample_pipeline):
    """Mock the training and prediction behavior of NoVacancyPipeline."""
    assert isinstance(
        sample_pipeline, NoVacancyPipeline
    ), "❌ sample_pipeline is not an instance of NoVacancyPipeline"

    sample_pipeline.fit = MagicMock(return_value=None)
    sample_pipeline.predict = MagicMock(return_value=[1, 0])
    sample_pipeline.predict_proba = MagicMock(return_value=[[0.1, 0.9], [0.8, 0.2]])
    return sample_pipeline


@pytest.fixture(scope="function")
def sample_processor():
    """Provide a valid sample NoVacancyDataProcessing instance."""
    processor = NoVacancyDataProcessing(
        variable_rename=VARIABLE_RENAME_MAP,
        month_abbreviation=MONTH_ABBREVIATION_MAP,
        vars_to_drop=VARS_TO_DROP,
        booking_map=BOOKING_MAP,
    )
    return processor


@pytest.fixture(scope="function")
def mock_processor(sample_processor):
    """Mock the behavior of NoVacancyDataProcessing."""
    sample_processor.fit = MagicMock(return_value=sample_processor)
    sample_processor.transform = MagicMock(
        return_value=(
            pd.DataFrame(
                {
                    "number_of_adults": [1, 2],
                    "number_of_children": [0, 1],
                    "month_of_reservation": ["Jan", "Feb"],
                    "day_of_week": ["Monday", "Tuesday"],
                }
            ),
            pd.Series([1, 0]),
        )
    )
    return sample_processor


@pytest.fixture(scope="function")
def mock_logger(mocker):
    """Mock the logger to suppress output during testing."""
    return mocker.patch("app.services.trainer.logger")


@pytest.fixture(scope="function")
def temp_booking_data_csv(booking_data):
    """Write booking_data to a temporary CSV file."""
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".csv"
    ) as temp_file:
        booking_data.to_csv(temp_file, index=False)
        temp_file.flush()  # Ensure data is written to the disk
        yield temp_file.name  # provide the temp file path to the test
    os.remove(temp_file.name)


@pytest.fixture(scope="session", autouse=True)
def cleanup_coverage_files(request):
    def cleanup():
        coverage_files = glob.glob(".coverage*")
        for file in coverage_files:
            try:
                os.remove(file)
                print(f"✅ Deleted: {file}")
            except OSError as e:
                print(f"Failed to delete {file}: {e}")

    # Register the cleanup function to run after all tests
    # request is a special fixture provided by pytest that allows dynamic test resource handling.
    request.addfinalizer(cleanup)


def pytest_sessionfinish(session, exitstatus):
    """
    Ensures .coverage files are removed after all tests are done.
    """
    print("\n🔄 Cleaning up .coverage files after all tests...")

    # Combine all coverage data
    # os.system("coverage combine")

    try:
        # Remove any lingering .coverage files
        coverage_files = glob.glob(".coverage*")
        time.sleep(3)
        for file in coverage_files:
            try:
                os.remove(file)
                print(f"✅ Deleted: {file}")
            except OSError as e:
                print(f"❌ Failed to delete {file}: {e}")
    except Exception as e:
        print(f"❌ pytest_sessionfinish encountered an error: {e}")


@pytest.fixture(scope="function")
def temp_pipeline_path(tmp_path):
    """Provide a temporary path for pipeline storage."""
    return tmp_path / "no_vacancy_pipeline.pkl"


@pytest.fixture(scope="function")
def test_db_conn():
    """
    Fixture for test DB connection and test table setup.
    """
    # Use try-except to ensure unit tests pass during CI even if the DB is unavailable.
    try:
        conn = psycopg2.connect(
            host=TEST_DB_HOST,
            port=TEST_DB_PORT,
            dbname=TEST_DB,
            user=TEST_DB_USER,
            password=TEST_DB_PASSWORD,
        )

    except psycopg2.OperationalError as e:
        pytest.skip(f"Test database not available: {e}")

    cursor = conn.cursor()

    cursor.execute(
        f"""
        DROP TABLE IF EXISTS {TEST_TABLE};
        CREATE TABLE {TEST_TABLE} (
            number_of_adults INTEGER,
            number_of_weekend_nights INTEGER

        DROP TABLE IF EXISTS csv_hashs (
            filename TEXT,
            file_hash TEXT
        );
        """
    )
    conn.commit()

    # Allow the test to access the connection
    yield conn

    cursor.execute(f"DROP TABLE IF EXISTS {TEST_TABLE};")
    conn.commit()
    cursor.close()
    conn.close()


# ------ Mock MLflow Fixtures ------


@pytest.fixture(autouse=True)
def mock_mlflow():
    with mock.patch("services.trainer.mlflow") as mock_ml:
        mock_ml.set_experiment.return_value = None
        mock_ml.start_run.return_value.__enter__.return_value = mock.Mock()
        mock_ml.log_params.return_value = None
        mock_ml.log_metric.return_value = None
        mock_ml.sklearn.log_model.return_value = None
        yield mock_ml


@pytest.fixture
def mock_mlflow_pipeline():
    """Mock MLflow pipeline for testing"""
    mock_pipeline = MagicMock()
    mock_pipeline.predict.return_value = np.array([0, 1, 0])
    mock_pipeline.predict_proba.return_value = np.array(
        [[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]]
    )
    return mock_pipeline


@pytest.fixture
def mock_mlflow_processor():
    """Mock MLflow processor for testing"""
    mock_processor = MagicMock()

    # Mock the transform method to return processed data
    def mock_transform(X, y=None):
        # Return a copy of X with some basic transformations
        X_processed = X.copy()
        if "booking_id" in X_processed.columns:
            X_processed.drop(columns=["booking_id"], inplace=True)
        return X_processed, y

    mock_processor.transform = mock_transform
    return mock_processor


@pytest.fixture
def mock_mlflow_client(mocker):
    """Mock MLflow client to avoid network calls"""
    mock_client = mocker.patch("mlflow.MlflowClient")
    mock_client.return_value.get_latest_versions.return_value = [
        MagicMock(run_id="test-run-id", version="1")
    ]

    # Mock mlflow.sklearn.load_model to return a mock pipeline
    mock_load_model = mocker.patch("mlflow.sklearn.load_model")
    mock_pipeline = MagicMock()
    mock_pipeline.predict.return_value = np.array([0, 1])
    mock_pipeline.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7]])
    mock_load_model.return_value = mock_pipeline

    # Mock mlflow.artifacts.download_artifacts
    mock_download = mocker.patch("mlflow.artifacts.download_artifacts")
    mock_download.return_value = "/tmp/mock_processor.pkl"

    # Mock joblib.load
    mock_joblib_load = mocker.patch("joblib.load")
    mock_processor = MagicMock()
    mock_processor.transform.return_value = (pd.DataFrame({"test": [1, 2]}), None)
    mock_joblib_load.return_value = mock_processor

    return mock_client


@pytest.fixture(autouse=True, scope="function")
def prevent_deployment_network_calls(
    request,
):  # request is a pytest fixture. provides info about the test.
    """
    Prevent MLflow network calls in deployment tests, unless the test
    has specific MLflow mocks.
    """
    # Skip the fixture if the testis already mocking MLflowArtifactLoader and its validator
    test_file = request.node.parent.name if hasattr(request.node, "parent") else ""
    if "test_inference_deployment" in test_file or "test_cd_pipeline" in test_file:
        yield
        return

    with mock.patch("mlflow.MlflowClient") as mock_client:
        mock_client.return_value.transition_model_version_stage.return_value = None
        mock_client.return_value.get_registered_model.return_value = MagicMock()
        yield mock_client


# ------ MLflowArtifactLoader Fixtures ------


@pytest.fixture
def mock_mlflow_artifact_setup():
    """Setup MLflow mocks for MLflowArtifactLoader tests."""
    with (
        patch("mlflow_utils.mlflow.set_tracking_uri") as mock_set_uri,
        patch("mlflow_utils.mlflow.MlflowClient") as mock_client_class,
    ):

        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        yield mock_set_uri, mock_client_class


@pytest.fixture
def mlflow_loader(mock_mlflow_artifact_setup):
    """Create MLflowArtifactLoader instance with mocked client."""
    return MLflowArtifactLoader()


@pytest.fixture
def mock_model_version():
    """Create a mock model version object for testing."""
    mock_version = MagicMock()
    mock_version.run_id = "test_run_id"
    mock_version.version = "7"
    mock_version.create_timestamp = 1234567890
    mock_version.description = "Test model version"
    mock_version.tags = {"validation_status": "approved"}
    mock_version.aliases = ["production", "latest"]
    return mock_version
