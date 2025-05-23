from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from schemas.bronze import RawData
from services.pipeline import NoVacancyPipeline
from services.trainer import (
    build_pipeline,
    evaluate_model,
    load_raw_data,
    preprocess_data,
    save_to_silver_db,
)
from sklearn.metrics import roc_auc_score


def test_load_raw_data_from_bronze(mocker, booking_data):
    # Arrange: Convert booking_data to list of mocked RawData objects
    mock_records = []

    # Iterate through dataframe rows
    for _, row in booking_data.iterrows():
        record = RawData()

        for col, val in row.items():
            # Convert column names to snake_case for SQLAlchemy attributes
            attr = col.lower().replace(" ", "_")
            setattr(record, attr, val)
        mock_records.append(record)

    # Create a mock BronzeSessionLocal
    mock_session = mocker.MagicMock()
    mock_session.query.return_value.all.return_value = mock_records

    # Act
    df_result = load_raw_data(mock_session)

    # Assert
    assert isinstance(df_result, pd.DataFrame)
    assert df_result.shape[0] == booking_data.shape[0]
    assert "booking_id" in df_result.columns
    assert "_sa_instance_state" not in df_result.columns


def test_preprocess_data(booking_data, sample_processor):
    # Arrange
    X = booking_data.drop(columns=["booking status"])
    y = booking_data["booking status"]

    # Action
    X_train, X_test, y_train, y_test = preprocess_data(X, y, sample_processor)

    # Assert that the processor's fit_transform method was called
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)

    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)


def test_save_to_silver_db(mocker):
    # Arrange
    X_mock = pd.DataFrame(
        [
            {
                "booking_id": f"id_{i}",
                "number_of_adults": 1,
                "number_of_children": 0,
                "number_of_weekend_nights": 1,
                "number_of_weekdays_nights": 2,
                "lead_time": 10,
                "type_of_meal": "Meal Plan 1",
                "car_parking_space": 0,
                "room_type": "Room_Type 1",
                "average_price": 100.0,
                **{
                    col: 0
                    for col in [
                        "is_type_of_meal_meal_plan_1",
                        "is_type_of_meal_meal_plan_2",
                        "is_type_of_meal_meal_plan_3",
                        "is_room_type_room_type_1",
                        "is_room_type_room_type_2",
                        "is_room_type_room_type_3",
                        "is_room_type_room_type_4",
                        "is_room_type_room_type_5",
                        "is_room_type_room_type_6",
                        "is_room_type_room_type_7",
                        "is_market_segment_type_online",
                        "is_market_segment_type_corporate",
                        "is_market_segment_type_complementary",
                        "is_market_segment_type_aviation",
                        "is_month_of_reservation_jan",
                        "is_month_of_reservation_feb",
                        "is_month_of_reservation_mar",
                        "is_month_of_reservation_apr",
                        "is_month_of_reservation_may",
                        "is_month_of_reservation_jun",
                        "is_month_of_reservation_aug",
                        "is_month_of_reservation_oct",
                        "is_month_of_reservation_nov",
                        "is_month_of_reservation_dec",
                        "is_day_of_week_monday",
                        "is_day_of_week_tuesday",
                        "is_day_of_week_wednesday",
                        "is_day_of_week_thursday",
                        "is_day_of_week_friday",
                        "is_day_of_week_saturday",
                    ]
                },
                "is_cancellation": 1,
            }
            for i in range(2)
        ]
    )

    y_mock = pd.Series([1, 0])

    mock_session = mocker.MagicMock()

    # Act
    save_to_silver_db(X_mock.copy(), y_mock, X_mock.copy(), y_mock, mock_session)

    # Assert
    assert mock_session.bulk_save_objects.call_count == 2
    assert mock_session.commit.called


def test_build_pipeline():
    pipe = build_pipeline()
    assert isinstance(pipe, NoVacancyPipeline)


# capsys is a builtin pytest fixture that captures output during the test
def test_evaluate_model(capsys):
    # Arrange
    mock_pipe = MagicMock()
    mock_pipe.predict_proba.return_value = np.array([[0.2, 0.8], [0.3, 0.7]])
    y_test = pd.Series([1, 0])

    # Act
    evaluate_model(mock_pipe, pd.DataFrame(), y_test)

    # capture printed output
    output = capsys.readouterr().out
    assert "AUC" in output
    assert mock_pipe.predict_proba.called
