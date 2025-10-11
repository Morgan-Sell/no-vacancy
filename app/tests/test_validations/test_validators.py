"""
Mission-critical tests for NoVacancyDataValidator.

These tests ensure data validation catches the most common production failures:
1. Schema drift (column changes)
2. Categorical mismatches (new/invalid values)
3. Data corruption (nulls, out-of-bounds values)
"""

import pandas as pd
import pytest

from validations.schemas import BRONZE_COLUMNS, EXPECTED_CATEGORIES
from validations.validators import DataQualityError, NoVacancyDataValidator


# ============================================
# FIXTURES
# ============================================


@pytest.fixture
def validator():
    """Create validator instance for tests."""
    return NoVacancyDataValidator()


@pytest.fixture
def valid_bronze_data(booking_data):
    """
    Create valid Bronze layer data for testing.
    Uses the booking_data fixture from conftest.py and renames columns
    to match the expected Bronze schema.
    """
    # booking_data fixture from conftest has spaces in column names
    # Need to rename the names to match BRONZE_COLUMNS
    df = booking_data.copy()

    # Rename colums to match Bronze DB schema
    df = df.rename(
        columns={
            "Booking_ID": "booking_id",
            "type of meal": "type_of_meal",
            "car parking space": "car_parking_space",
            "room type": "room_type",
            "repeated": "is_repeat_guest",
            "P-C": "num_previous_cancellations",
            "P-not-C": "num_previous_bookings_not_canceled",
        }
    )

    return df


@pytest.fixture
def valid_silver_data(preprocessed_booking_data):
    """
    Create valid Silver layer data for testing.
    Uses the preprocessed_booking_data from conftest.py.
    """
    X_prcsd, y_prcsd = preprocessed_booking_data

    # Add the target variable back
    df = X_prcsd.copy()
    df["is_cancellation"] = y_prcsd

    return df


# ============================================
# BRONZE LAYER TESTS
# ============================================


class TestBronzeValidation:
    """Test Bronze layer validation catches critical issues."""

    def test_valid_bronze_data_passes(self, validator, valid_bronze_data):
        """Valid Bronze data should pass validations."""
        results = validator.validate_bronze_data(valid_bronze_data)

        assert results["success"] is True
        assert results["statistics"]["unsuccessful_expectations"] == 0

    def test_missing_column_fails(self, validator, valid_bronze_data):
        """Missing required column should fail validation."""
        # Remove critical column
        invalid_data = valid_bronze_data.drop(columns=["booking_status"])

        results = validator.validate_bronze_data(invalid_data)

        assert results["success"] is False

        # Confirm that schema validation failed
        failed_types = [
            r.expectation_config.type for r in results["failed_expectations"]
        ]
        assert "expect_table_columns_to_match_ordered_list" in failed_types

    def test_reordered_columns_fails(self, validator, valid_bronze_data):
        """Reordered columns should fail schema validation."""
        # Reorder columns
        cols = list(valid_bronze_data.columns)
        reordered_cols = [cols[1], cols[0]] + cols[2:]  # Swap first two
        invalid_data = valid_bronze_data[reordered_cols]

        results = validator.validate_bronze_data(invalid_data)

        assert results["success"] is False
        # Confirm column order failed
        failed_types = [
            r.expectation_config.type for r in results["failed_expectations"]
        ]
        assert "expect_table_columns_to_match_ordered_list" in failed_types

    def test_duplicate_booking_ids_fails(self, validator, valid_bronze_data):
        """Duplicate booking IDs should fail uniqueness check."""
        invalid_data = valid_bronze_data.copy()

        # Create a duplicate by coping the first's rows booking_id
        invalid_data.loc[1, "booking_id"] = invalid_data.loc[0, "booking_id"]

        results = validator.validate_bronze_data(invalid_data)

        assert results["success"] is False
        failed_types = [
            r.expectation_config.type for r in results["failed_expectations"]
        ]
        assert "expect_column_values_to_be_unique" in failed_types

    def test_null_booking_id_fails(self, validator, valid_bronze_data):
        """Null booking_id should fail validation."""
        invalid_data = valid_bronze_data.copy()
        invalid_data.loc[0, "booking_id"] = None

        results = validator.validate_bronze_data(invalid_data)

        assert results["success"] is False
        failed_types = [
            r.expectation_config.type for r in results["failed_expectations"]
        ]
        assert "expect_column_values_to_not_be_null" in failed_types

    def test_invalid_meal_plan_fails(self, validator, valid_bronze_data):
        """Invalid room type should fail validation."""
        invalid_data = valid_bronze_data.copy()
        invalid_data.loc[0, "room_type"] = "Meal Plan 777"  # Invalid

        results = validator.validate_bronze_data(invalid_data)

        assert results["success"] is False
        failed_types = [
            r.expectation_config.type for r in results["failed_expectations"]
        ]
        assert "expect_column_values_to_be_in_set" in failed_types

    def test_invalid_room_type_fails(self, validator, valid_bronze_data):
        """Invalid room type should fail validation."""
        invalid_data = valid_bronze_data.copy()
        invalid_data.loc[0, "room_type"] = "Room_Type 369"  # Invalid

        results = validator.validate_bronze_data(invalid_data)

        assert results["success"] is False

    def test_invalid_market_segment_fails(self, validator, valid_bronze_data):
        """Invalid market segment should fail validation"""
        invalid_data = valid_bronze_data.copy()
        invalid_data.loc[0, "market_segment_type"] = "DarkWeb"

        results = validator.validate_bronze_data(invalid_data)

        assert results["success"] is False

    def test_out_of_bounds_adults_fails(self, validator, valid_bronze_data):
        """Out-of-bounds number of adults should fail validation."""
        invalid_data = valid_bronze_data.copy()
        invalid_data.loc[0, "number_of_adults"] = 500

        results = validator.validate_bronze_data(invalid_data)

        assert results["success"] is False
        failed_types = [
            r.expectation_config.type for r in results["failed_expectations"]
        ]
        assert "expect_column_values_to_be_between" in failed_types

    def test_negative_price_fails(self, validator, valid_bronze_data):
        """Negative price should fail validation."""
        invalid_data = valid_bronze_data.copy()
        invalid_data.loc[0, "average_price"] = -100  # Negative price!

        results = validator.validate_bronze_data(invalid_data)

        assert results["success"] is False

    def test_excessive_lead_time_fails(self, validator, valid_bronze_data):
        """Excessive lead time should fail validation."""
        invalid_data = valid_bronze_data.copy()
        invalid_data.loc[0, "lead_time"] = 1000  # 3+ years advance booking!

        results = validator.validate_bronze_data(invalid_data)

        assert results["success"] is False

    def test_too_few_rows_fails(self, validator, valid_bronze_data):
        """Too few rows should fail row count validation."""
        # Take only first 5 rows (min is 1000)
        invalid_data = valid_bronze_data.head(5)

        results = validator.validate_bronze_data(invalid_data)

        assert results["success"] is False
        failed_types = [
            r.expectation_config.type for r in results["failed_expectations"]
        ]
        assert "expect_table_row_count_to_be_between" in failed_types


# ============================================
# SILVER LAYER TESTS
# ============================================


class TestSilverValidation:
    """Test Silver layer validation catches transformation issues."""

    def test_valid_silver_data_passes(self, validator, valid_silver_data):
        """Valid Silver data should pass all validations."""
        results = validator.validate_silver_data(valid_silver_data)

        assert results["success"] is True

    def test_invalid_month_fails(self, validator, valid_silver_data):
        """Invalid month value should fail validation."""
        invalid_data = valid_silver_data.copy()
        invalid_data.loc[0, "month_of_reservation"] = "InvalidMonth"

        results = validator.validate_silver_data(invalid_data)

        assert results["success"] is False
        failed_types = [
            r.expectation_config.type for r in results["failed_expectations"]
        ]
        assert "expect_column_values_to_be_in_set" in failed_types

    def test_invalid_day_fails(self, validator, valid_silver_data):
        """Invalid day of week should fail validation."""
        invalid_data = valid_silver_data.copy()
        invalid_data.loc[0, "day_of_week"] = "Funday"  # Not a real day!

        results = validator.validate_silver_data(invalid_data)

        assert results["success"] is False

    def test_non_binary_target_fails(self, validator, valid_silver_data):
        """Non-binary target value should fail validation."""
        invalid_data = valid_silver_data.copy()
        invalid_data.loc[0, "is_cancellation"] = 5  # Should be 0 or 1!

        results = validator.validate_silver_data(invalid_data)

        assert results["success"] is False
        failed_types = [
            r.expectation_config.type for r in results["failed_expectations"]
        ]
        assert "expect_column_values_to_be_in_set" in failed_types

    def test_string_target_fails(self, validator, valid_silver_data):
        """String target value (not transformed) should fail validation."""
        invalid_data = valid_silver_data.copy()
        invalid_data.loc[0, "is_cancellation"] = "Canceled"  # Should be integer!

        results = validator.validate_silver_data(invalid_data)

        assert results["success"] is False

    def test_raw_booking_status_still_present_fails(self, validator, valid_silver_data):
        """Presence of raw booking_status column should fail validation."""
        invalid_data = valid_silver_data.copy()
        invalid_data["booking_status"] = "Canceled"  # Should be dropped!

        results = validator.validate_silver_data(invalid_data)

        assert results["success"] is False
        failed_types = [
            r.expectation_config.type for r in results["failed_expectations"]
        ]
        assert "expect_table_columns_to_not_contain_column_list" in failed_types

    def test_raw_date_column_still_present_fails(self, validator, valid_silver_data):
        """Presence of raw date column should fail validation."""
        invalid_data = valid_silver_data.copy()
        invalid_data["date_of_reservation"] = "10/2/2015"  # Should be dropped!

        results = validator.validate_silver_data(invalid_data)

        assert results["success"] is False


# ============================================
# MODEL INPUT TESTS
# ============================================


class TestModelInputValidation:
    """Test Model Input validation catches feature mismatches."""

    def test_valid_ohe_columns_pass(self, validator):
        """Valid OneHotEncoded columns should pass."""
        valid_data = pd.DataFrame(
            {
                "booking_id": ["BK001", "BK002", "BK003"],
                "number_of_adults": [2, 1, 3],
                "is_type_of_meal_meal_plan_1": [1, 0, 0],
                "is_type_of_meal_meal_plan_2": [0, 1, 0],
                "is_room_type_room_type_1": [1, 0, 1],
            }
        )

        results = validator.validate_model_input(valid_data)

        assert results["success"] is True

    def test_non_binary_ohe_fails(self, validator):
        """Non-binary OHE column should fail validation."""
        invalid_data = pd.DataFrame(
            {
                "booking_id": ["BK001", "BK002", "BK003"],
                "number_of_adults": [2, 1, 3],
                "is_type_of_meal_meal_plan_1": [1, 2, 0],  # 2 is invalid!
            }
        )

        results = validator.validate_model_input(invalid_data)

        assert results["success"] is False
        failed_types = [
            r.expectation_config.type for r in results["failed_expectations"]
        ]
        assert "expect_column_values_to_be_in_set" in failed_types

    def test_float_ohe_fails(self, validator):
        """Float values in OHE column should fail validation."""
        invalid_data = pd.DataFrame(
            {
                "booking_id": ["BK001", "BK002", "BK003"],
                "number_of_adults": [2, 1, 3],
                "is_type_of_meal_meal_plan_1": [1.5, 0, 0],  # Float is invalid!
            }
        )

        results = validator.validate_model_input(invalid_data)

        assert results["success"] is False

    def test_no_ohe_columns_logs_warning(self, validator, caplog):
        """No OHE columns should log a warning."""
        data_without_ohe = pd.DataFrame(
            {
                "booking_id": ["BK001", "BK002", "BK003"],
                "number_of_adults": [2, 1, 3],
                "average_price": [100, 150, 120],
            }
        )

        results = validator.validate_model_input(data_without_ohe)

        # Should still pass (no OHE columns to validate)
        assert results["success"] is True

        # Check that warning was logged
        assert "No OneHotEncoded columns found" in caplog.text


# ============================================
# CONVENIENCE METHOD TESTS
# ============================================


class TestValidateAndRaise:
    """Test validate_and_raise convenience method."""

    def test_raises_on_invalid_bronze_data(self, validator, valid_bronze_data):
        """Should raise DataQualityError on invalid data."""
        invalid_data = valid_bronze_data.copy()
        invalid_data.loc[0, "booking_id"] = None  # Invalid!

        with pytest.raises(DataQualityError, match="Bronze layer validation failed"):
            validator.validate_and_raise(invalid_data, layer="bronze")

    def test_raises_on_invalid_silver_data(self, validator, valid_silver_data):
        """Should raise DataQualityError on invalid Silver data."""
        invalid_data = valid_silver_data.copy()
        invalid_data.loc[0, "is_cancellation"] = 99  # Invalid!

        with pytest.raises(DataQualityError, match="Silver layer validation failed"):
            validator.validate_and_raise(invalid_data, layer="silver")

    def test_returns_results_on_valid_data(self, validator, valid_bronze_data):
        """Should return results dict on valid data."""
        results = validator.validate_and_raise(valid_bronze_data, layer="bronze")

        assert results["success"] is True
        assert "statistics" in results
        assert "failed_expectations" in results

    def test_raises_on_unknown_layer(self, validator, valid_bronze_data):
        """Should raise ValueError on unknown layer."""
        with pytest.raises(ValueError, match="Unknown layer"):
            validator.validate_and_raise(valid_bronze_data, layer="invalid_layer")


# ============================================
# INTEGRATION TESTS
# ============================================


class TestValidatorIntegration:
    """Test validator works with real preprocessing pipeline."""

    def test_bronze_to_silver_workflow(
        self, validator, valid_bronze_data, sample_processor
    ):
        """Test validation works through Bronze -> Silver transformation"""
        # Validate Bronze
        bronze_results = validator.validate_bronze_data(valid_bronze_data)
        assert bronze_results["success"] is True

        # Transform to Silver
        X = valid_bronze_data.drop(columns=["booking_status"])
        y = valid_bronze_data["booking_status"]
        X_silver, y_silver = sample_processor.fit_transform(X, y)

        # Add target back
        silver_data = X_silver.copy()
        silver_data["is_cancellation"] = y_silver

        # Validate Silver
        silver_results = validator.validate_silver_data(silver_data)
        assert silver_results["success"] is True

    def test_end_to_end_validation(
        self, validator, valid_bronze_data, preprocessed_booking_data
    ):
        """Test end-to-end validation from Bronze to Model Input."""
        # 1. Validate Bronze
        bronze_results = validator.validate_bronze_data(valid_bronze_data)
        assert bronze_results["success"] is True

        # 2. Get preprocessed data (Silver)
        X_processed, y_processed = preprocessed_booking_data
        silver_data = X_processed.copy()
        silver_data["is_cancellation"] = y_processed

        # 3. Validate Silver
        silver_results = validator.validate_silver_data(silver_data)
        assert silver_results["success"] is True

        # 4. Validate Model Input (just features, no target)
        model_results = validator.validate_model_input(X_processed)
        assert model_results["success"] is True
