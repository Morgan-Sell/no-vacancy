import os
import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from feature_engine.encoding import OneHotEncoder
from feature_engine.imputation import CategoricalImputer
from sklearn.ensemble import RandomForestClassifier

from app.services.pipeline import NoVacancyPipeline
from app.services.preprocessing import NoVacancyDataProcessing


@pytest.fixture(scope="function")
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
            "5/20/2017",
            "4/11/2018",
            "3/15/2019",
            "12/25/2020",
            "8/19/2016",
            "9/12/2020",
            "7/8/2021",
            "1/15/2022",
            "6/25/2020",
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
def sample_pipeline():
    processor = NoVacancyDataProcessing(
        variable_rename={},
        month_abbreviation={},
        vars_to_drop=[],
        booking_map={},
    )
    imputer = CategoricalImputer()
    encoder = OneHotEncoder()
    estimator = RandomForestClassifier()

    return NoVacancyPipeline(processor, imputer, encoder, estimator)


@pytest.fixture(scope="function")
def mock_read_csv(mocker, booking_data):
    """Mock pandas read_csv to return booking_data."""
    return mocker.patch("pandas.read_csv", return_value=booking_data)


@pytest.fixture(scope="function")
def mock_pipeline(mocker, sample_pipeline):
    """Mock the NoVacancyPipeline to avoid actual model training."""
    mock_pipeline = mocker.patch(
        "app.services.trainer.NoVacancyPipeline", return_value=sample_pipeline
    )
    mock_pipeline.fit = MagicMock(return_value=None)
    mock_pipeline.predict_proba = MagicMock(return_value=[[0.1, 0.9], [0.7, 0.3]])
    return mock_pipeline


@pytest.fixture(scope="function")
def mock_logger(mocker):
    """Mock the logger to suppress output during testing."""
    return mocker.patch("app.services.trainer.logger")


@pytest.fixture(scope="function")
def temp_booking_data_csv(booking_data):
    """Write booking_data to a temporary CSV file."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
        booking_data.to_csv(temp_file, index=False)
        temp_file.flush()  # Ensure data is written to the disk
        yield temp_file.name  # provide the temp file path to the test
    os.remove(temp_file.name)
