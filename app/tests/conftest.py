import glob
import os
import random
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import psycopg2
import pytest
from config import TEST_DB, TEST_DB_HOST, TEST_DB_PASSWORD, TEST_DB_PORT, TEST_DB_USER
from feature_engine.encoding import OneHotEncoder
from feature_engine.imputation import CategoricalImputer
from services import (
    BOOKING_MAP,
    DATA_PATHS,
    IMPUTATION_METHOD,
    MONTH_ABBREVIATION_MAP,
    VARIABLE_RENAME_MAP,
    VARS_TO_DROP,
    VARS_TO_IMPUTE,
    VARS_TO_OHE,
)
from services.pipeline import NoVacancyPipeline
from services.pipeline_management import PipelineManagement
from services.preprocessing import NoVacancyDataProcessing
from sklearn.ensemble import RandomForestClassifier
from tests import TEST_TABLE

# -- Helper Functions --


@pytest.fixture(scope="session")
def booking_data():
    data = {
        "Booking_ID": [f"INN0000{i}" for i in range(1, 21)],
        "number of adults": [
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
        "number of children": [
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
        "number of weekend nights": [
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
        "number of week nights": [
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
        "lead time": [
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
        "market segment type": [
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
        "average price": [
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
        "special requests": [
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
        "date of reservation": [
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
        "booking status": [
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

    pipeline = NoVacancyPipeline(imputer, encoder, estimator)
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
def pm(temp_pipeline_path):
    """
    Use fixture to instantiate DataManagement to follow DRY
    principle and enable easier code changes.
    """
    return PipelineManagement(pipeline_path=str(temp_pipeline_path))


@pytest.fixture(scope="function")
def trained_pipeline_and_processor(booking_data, tmp_path):
    # Preprocessing
    processor = NoVacancyDataProcessing(
        variable_rename=VARIABLE_RENAME_MAP,
        month_abbreviation=MONTH_ABBREVIATION_MAP,
        vars_to_drop=VARS_TO_DROP,
        booking_map=BOOKING_MAP,
    )
    X = booking_data.drop(columns=["booking status"])
    y = booking_data["booking status"]
    X_train_prcsd, y_train_prcsd = processor.fit_transform(X, y)

    # Build + train pipeline
    imputer = CategoricalImputer(
        imputation_method=IMPUTATION_METHOD, variables=VARS_TO_IMPUTE
    )
    encoder = OneHotEncoder(variables=VARS_TO_OHE)
    clsfr = RandomForestClassifier()
    search_space = {
        "n_estimators": [20, 50, 100, 200],
        "max_features": ["log2", "sqrt"],
        "max_depth": [1, 3, 5],
        "min_samples_split": [2, 5, 10],
    }
    pipe = NoVacancyPipeline(imputer, encoder, clsfr)
    pipe.fit(X_train_prcsd, y_train_prcsd, search_space)

    # Save the trained pipeline and processor
    temp_pipeline_path = tmp_path / DATA_PATHS["model_save_path"]
    pm = PipelineManagement(pipeline_path=temp_pipeline_path)
    pm.save_pipeline(pipe, processor)

    # Move the model artifacts to the app path
    app_path = Path(DATA_PATHS["model_save_path"])
    app_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(temp_pipeline_path, app_path)

    # In case there's a need to return the variables
    return pipe, processor, pm


@pytest.fixture(scope="function")
def test_db_conn():
    """
    Fixture for test DB connection and test table setup.
    """
    conn = psycopg2.connect(
        host=TEST_DB_HOST,
        port=TEST_DB_PORT,
        dbname=TEST_DB,
        user=TEST_DB_USER,
        password=TEST_DB_PASSWORD,
    )

    cursor = conn.cursor()

    cursor.execute(
        f"""
        DROP TABLE IF EXISTS {TEST_TABLE};
        CREATE TABLE {TEST_TABLE} (
            number_of_adults INTEGER,
            number_of_weekend_nights INTEGER
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
