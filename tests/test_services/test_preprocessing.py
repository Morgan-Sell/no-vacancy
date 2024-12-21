import pytest

from app.services.config_services import (MONTH_ABBREVIATION_MAP,
                                          VARIABLE_RENAME_MAP)
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
    )

    # Action & Assert
    assert transformer._to_snake_case(input_str) == expected_output


def test_convert_columns_to_snake_case(booking_data):
    # Arrange
    transformer = NoVacancyDataProcessing(
        variable_rename={},
        month_abbreviation={},
        vars_to_drop=[],
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
        "market_segment",
        "type",
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
    )

    # Action
    result_df = transformer.transform(booking_data)

    # Assert
    assert result_df.shape[0] == booking_data.shape[0]
    # Add 2 for the new columns "month_of_reservation" and "day_of_week"
    assert result_df.shape[1] == booking_data.shape[1] - len(vars_to_drop) + 2

    # Check that the columns have been dropped
    assert "booking_id" not in result_df.columns
    assert "date_of_reservation" not in result_df.columns

    # Check that "month_of_reservation" was properly extracted
    assert list(result_df["month_of_reservation"]) == [
        "Oct",
        "Nov",
        "Feb",
        "May",
        "Apr",
        "Mar",
        "Dec",
        "Aug",
        "Sep",
        "Jul",
    ]

    # Check that "day_of_week" was properly extracted
    assert list(result_df["day_of_week"]) == [
        "Saturday",
        "Sunday",
        "Monday",
        "Thursday",
        "Friday",
        "Monday",
        "Friday",
        "Saturday",
        "Tuesday",
        "Wednesday",
    ]
