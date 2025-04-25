import glob
import os
import random
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path
from random import random
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# 'app' is not required because pytest automatically adds the root directory to sys.path
# This capability is configured in pyproject.toml.
from db.postgres import (
    BronzeSessionLocal,
    GoldSessionLocal,
    SilverSessionLocal,
    init_all_databases,
)
from feature_engine.encoding import OneHotEncoder
from feature_engine.imputation import CategoricalImputer
from schemas.bronze import RawData
from schemas.gold import TestResults, TrainResults, ValidationResults
from schemas.silver import TrainData, ValidationTestData
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

from tests import (
    transform_booking_data_to_bronze_db_format,
    transform_booking_data_to_silver_db_format,
    get_db_model_column_names,
)


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


@pytest.fixture(scope="session")
def setup_test_dbs(booking_data):
    init_all_databases()

    # Prepare booking_data for Bronze db



    # Seed Bronze database
    with BronzeSessionLocal() as session:
        # Ensure the table is empty before seeding
        session.query(RawData).delete()
        session.commit()

        # Transform booking_data to match the Bronze DB schema
        # bronze_data = transform_booking_data_to_bronze_db_format(booking_data.copy())


        # Seed the Bronze database with booking_data
        for _, row in booking_data.iterrows():
            session.add(
                RawData(
                    booking_id=row["Booking_ID"],
                    number_of_adults=row["number of adults"],
                    number_of_children=row["number of children"],
                    number_of_weekend_nights=row["number of weekend nights"],
                    number_of_week_nights=row["number of week nights"],
                    type_of_meal=row["type of meal"],
                    car_parking_space=row["car parking space"],
                    room_type=row["room type"],
                    lead_time=row["lead time"],
                    market_segment_type=row["market segment type"],
                    is_repeat_guest=row["repeated"],
                    num_previous_cancellations=row["P-C"],
                    num_previous_bookings_not_canceled=row["P-not-C"],
                    average_price=row["average price"],
                    special_requests=row["special requests"],
                    date_of_reservation=datetime.strptime(
                        row["date of reservation"], "%m/%d/%Y"
                    ).date(),
                    booking_status=row["booking status"],
                )
            )
        session.commit()

        # Prepare booking_data for Silver db (split half into train, half into validate/test)
        silver_data = transform_booking_data_to_silver_db_format(booking_data.copy())
        mid = len(silver_data) // 2
        silver_train_rows = silver_data.head(mid).copy()
        silver_test_rows = silver_data.tail(mid).copy()


        with SilverSessionLocal() as session:
            # Ensure the tables are empty before seeding
            session.query(TrainData).delete()
            session.query(ValidationTestData).delete()
            session.commit()

            # Seed the Silver database with transformed booking_data
            session.bulk_save_objects(
                [TrainData(**row) for row in silver_train_rows.to_dict(orient="records")]
            )

            session.bulk_save_objects(
                [ValidationTestData(**row) for row in silver_test_rows.to_dict(orient="records")]
            )
            session.commit()

        # Seed Gold database
        with GoldSessionLocal() as session:
            # Ensure the tables are empty before seeding
            session.query(TrainResults).delete()
            session.query(ValidationResults).delete()
            session.query(TestResults).delete()
            session.commit()

            # Seed the Gold database with booking_data
            for _, row in booking_data.iterrows():
                is_not_canceled = row["booking status"] == "Not_Canceled"
                if is_not_canceled:
                    prob = round(random.uniform(0.70, 0.99), 2)
                else:
                    prob = round(random.uniform(0.01, 0.30), 2)

                gold_kwargs = dict(
                    booking_id=row["Booking_ID"],
                    prediction=int(is_not_canceled),
                    probability_not_canceled=prob,
                    probability_canceled=1 - prob,
                    created_at=datetime.today().date(),
                )
                session.add(TrainResults(**gold_kwargs))
                session.add(ValidationResults(**gold_kwargs))
                session.add(TestResults(**gold_kwargs))
            session.commit()
