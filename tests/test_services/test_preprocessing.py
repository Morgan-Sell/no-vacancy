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
    vars_to_drop = ["booking_id", "date_of_reservation"]
    transformer = NoVacancyDataProcessing(
        variable_rename=VARIABLE_RENAME_MAP,
        month_abbreviation=MONTH_ABBREVIATION_MAP,
        vars_to_drop=vars_to_drop,
    )

    # Action & Assert
    assert transformer._to_snake_case(input_str) == expected_output
