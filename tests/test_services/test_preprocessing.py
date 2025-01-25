import pandas as pd
import pytest

from app.services import (
    BOOKING_MAP,
    MONTH_ABBREVIATION_MAP,
    VARIABLE_RENAME_MAP,
)
from app.services.preprocessing import NoVacancyDataProcessing


@pytest.mark.parametrize(
    "input_str, expected_output",
    [
        ("JerrySeinfeld", "jerry_seinfeld"),  # CamelCase
        ("Elaine Benes", "elaine_benes"),  # Space between names
        ("Cosmo Kramer", "cosmo_kramer"),  # Space-separated names
        ("George Costanza", "george_costanza"),  # Space-separated names
        ("NoSoup ForYou", "no_soup_for_you"),  # Mixed camelCase and spaces
        (
            "Festivus For the Rest of Us",
            "festivus_for_the_rest_of_us",
        ),  # Multiple spaces
        ("ManHands Example", "man_hands_example"),  # CamelCase reference
        ("Soup-Nazi!", "soup_nazi"),  # Special characters
        ("shrinkage", "shrinkage"),  # Single word
        ("yada     yada     yada", "yada_yada_yada"),  # Multiple spaces between words
    ],
)
def test_to_snake_case(input_str, expected_output):
    # Arrange
    transformer = NoVacancyDataProcessing(
        variable_rename={},
        month_abbreviation={},
        vars_to_drop=[],
        booking_map={},
    )

    # Action & Assert
    assert transformer._to_snake_case(input_str) == expected_output


def test_convert_columns_to_snake_case(booking_data):
    # Arrange
    transformer = NoVacancyDataProcessing(
        variable_rename={},
        month_abbreviation={},
        vars_to_drop=[],
        booking_map={},
    )

    # Action
    result_df = transformer._convert_columns_to_snake_case(booking_data)

    # Assert
    expected_columns = [
        "booking_id",
        "number_of_adults",
        "number_of_children",
        "number_of_weekend_nights",
        "number_of_week_nights",
        "type_of_meal",
        "car_parking_space",
        "room_type",
        "lead_time",
        "market_segment_type",
        "repeated",
        "p_c",
        "p_not_c",
        "average_price",
        "special_requests",
        "date_of_reservation",
        "booking_status",
    ]

    assert list(result_df.columns) == expected_columns


def test_no_vacancy_data_processing_transform(booking_data):
    # Arrange
    vars_to_drop = ["booking_id", "date_of_reservation"]

    transformer = NoVacancyDataProcessing(
        variable_rename=VARIABLE_RENAME_MAP,
        month_abbreviation=MONTH_ABBREVIATION_MAP,
        vars_to_drop=vars_to_drop,
        booking_map=BOOKING_MAP,
    )

    X = booking_data.drop(columns=["booking status"])
    y = booking_data["booking status"]

    # Action
    transformer.fit(X, y)
    X_tr, y_tr = transformer.transform(X, y)

    # Assert
    assert X_tr.shape[0] == booking_data.shape[0]
    assert y_tr.shape[0] == booking_data.shape[0]
    # Add 2 for the new columns "month_of_reservation" and "day_of_week"
    # Subtract 1 because "booking status" is removed.
    assert X_tr.shape[1] == booking_data.shape[1] - len(vars_to_drop) + 2 - 1

    # Check that the columns have been dropped
    assert "Booking_ID" not in X_tr.columns
    assert "date of reservation" not in X_tr.columns

    # Check that "month_of_reservation" was properly extracted
    expected_months = (
        pd.to_datetime(booking_data["date of reservation"]).dt.strftime("%b").tolist()
    )
    assert all(
        month == expected_month
        for month, expected_month in zip(
            X_tr["month_of_reservation"].tolist(), expected_months
        )
    )

    # Check that "day_of_week" was properly extracted
    expected_weekdays = (
        pd.to_datetime(booking_data["date of reservation"]).dt.strftime("%A").tolist()
    )
    assert all(
        weekday == expected_weekday
        for weekday, expected_weekday in zip(
            X_tr["day_of_week"].tolist(), expected_weekdays
        )
    )

    # Confirm y_tr only contains 0s and 1s
    unique_values = set(y_tr.unique())
    assert unique_values.issubset({0, 1})
