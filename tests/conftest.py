import pandas as pd
import pytest
from xgboost import XGBClassifier

from app.services.preprocessing import NoVacancyDataProcessing
from app.services.pipeline import NoVacancyPipeline

from feature_engine.encoding import OneHotEncoder
from feature_engine.imputation import CategoricalImputer


@pytest.fixture(scope="function")
def booking_data():
    data = {
        "Booking_ID": [f"INN0000{i}" for i in range(1, 11)],
        "number_of_adults": [1, 1, 2, 1, 1, 2, 2, 1, 2, 1],
        "number_of_children": [1, 0, 1, 0, 0, 2, 1, 1, 0, 2],
        "number_of_weekend_nights": [2, 1, 1, 0, 1, 2, 0, 1, 1, 0],
        "number_of_week_nights": [5, 3, 3, 2, 2, 4, 3, 5, 2, 1],
        "type_of_meal": [
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
        ],
        "car_parking_space": [0, 0, 0, 0, 0, 1, 1, 0, 0, 1],
        "room_type": [
            "Room_Type 1",
            "Room_Type 3",
            "Room_Type 3",
            "Room_Type 2",
            "Room_Type 2",
            "Room_Type 1",
            "Room_Type 3",
            "Room_Type 2",
            "Room_Type 1",
            "Room_Type 1",
        ],
        "lead_time": [224, 5, 1, 211, 48, 150, 35, 60, 20, 10],
        "market_segment": [
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
        ],
        "type": ["P-C"] * 10,
        "repeated": [0] * 10,
        "P-C": [0] * 10,
        "P-not-C": [0] * 10,
        "average_price": [
            88.00,
            106.68,
            50.00,
            100.00,
            77.00,
            120.00,
            85.50,
            90.00,
            60.00,
            110.00,
        ],
        "special_requests": [0, 1, 0, 1, 0, 0, 1, 1, 0, 1],
        "date_of_reservation": [
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
        ],
    }
    df = pd.DataFrame(data)
    return df


@pytest.fixture(scope="function")
def sample_pipeline():
    processor = NoVacancyDataProcessing(
        variable_rename={}, month_abbreviation={}, vars_to_drop=[]
    )
    imputer = CategoricalImputer()
    encoder = OneHotEncoder()
    estimator = XGBClassifier()

    return NoVacancyPipeline(processor, imputer, encoder, estimator)
