"""
Mission-critical tests for NoVacancyDataValidator.

These tests ensure data validation catches the most common production failures:
1. Schema drift (column changes)
2. Categorical mismatches (new/invalid values)
3. Data corruption (nulls, out-of-bounds values)
"""

import logging
from pprint import pprint
import pandas as pd
import pytest

from validations.schemas import BRONZE_COLUMNS, EXPECTED_CATEGORIES
from validations.validators import DataQualityError, NoVacancyDataValidator
from scripts.import_csv_to_postgres import normalize_column_name
from services import (
    VARIABLE_RENAME_MAP,
    MONTH_ABBREVIATION_MAP,
    BOOKING_MAP,
    VARS_TO_DROP,
)


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

    Simulates ETL normalization by apply
    scripts.import_csv_to_postgres.normalize_column_name
    on booking_data fixture from conftest.py
    """
    # booking_data fixture from conftest has spaces in column names
    # Need to rename the names to match BRONZE_COLUMNS
    df = booking_data.copy()

    # Rename colums to match Bronze DB schema
    df.columns = [normalize_column_name(col) for col in df.columns]

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
        mismatch = results["failed_expectations"][0]["result"]["details"]["mismatched"][
            0
        ]
        assert "booking_status" == mismatch["Expected"]

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
        invalid_data = pd.DataFrame(columns=valid_bronze_data.columns)

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
        invalid_data["booking_status"] = ["Canceled"] * invalid_data.shape[
            0
        ]  # Should be dropped!

        results = validator.validate_silver_data(invalid_data)

        assert results["success"] is False
        assert "booking_status" in results["error"]

    def test_raw_date_column_still_present_fails(self, validator, valid_silver_data):
        """Presence of raw date column should fail validation."""
        invalid_data = valid_silver_data.copy()
        invalid_data["date_of_reservation"] = (
            "10/2/2015" * invalid_data.shape[0]
        )  # Should be dropped!

        results = validator.validate_silver_data(invalid_data)

        assert results["success"] is False
        assert "date_of_reservation" in results["error"]


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
    """Integration tests for multi-layer validation workflow."""

    def test_end_to_end_validation(
        self, validator, valid_bronze_data, preprocessed_booking_data
    ):
        """Test end-to-end validation from Bronze to Silver layers.

        Note: Model input validation removed - OHE and scaling happen inside
        the sklearn pipeline during training, which validates schema implicitly.
        """
        # 1. Validate Bronze layer
        bronze_results = validator.validate_bronze_data(valid_bronze_data)
        assert bronze_results["success"] is True
        assert bronze_results["statistics"]["unsuccessful_expectations"] == 0

        # 2. Get preprocessed data from fixture (Silver layer)
        X_processed, y_processed = preprocessed_booking_data
        silver_data = X_processed.copy()
        silver_data["is_cancellation"] = y_processed

        # 3. Validate Silver layer
        silver_results = validator.validate_silver_data(silver_data)
        assert silver_results["success"] is True
        assert silver_results["statistics"]["unsuccessful_expectations"] == 0

    def test_bronze_to_silver_workflow(
        self, validator, valid_bronze_data, preprocessed_booking_data
    ):
        """Test that Bronze data transforms to valid Silver data."""

        # 1. Validate Bronze
        bronze_results = validator.validate_bronze_data(valid_bronze_data)
        assert bronze_results["success"] is True

        # 2. Use the existing preprocessed_booking_data fixture
        # (which already transforms Bronze → Silver correctly)
        X_silver, y_silver = preprocessed_booking_data

        # 3. Reconstruct Silver DataFrame for validation
        silver_data = X_silver.copy()
        silver_data["is_cancellation"] = y_silver

        # 4. Validate Silver
        silver_results = validator.validate_silver_data(silver_data)
        assert silver_results["success"] is True

    def test_validation_catches_issues_at_each_layer(
        self, validator, valid_bronze_data
    ):
        """Verify each validation layer catches appropriate issues."""

        # Bronze catches schema issues
        invalid_bronze = valid_bronze_data.copy()
        invalid_bronze = invalid_bronze.drop(columns=["booking_status"])

        bronze_results = validator.validate_bronze_data(invalid_bronze)
        assert bronze_results["success"] is False

        # Silver catches transformation issues
        from services.preprocessing import NoVacancyDataProcessing

        # Initialize preprocessor with required parameters
        variable_rename = {
            "number_of_adults": "number_of_adults",
            "number_of_children": "number_of_children",
            "date_of_reservation": "date_of_reservation",
            # ... rest of mappings
        }

        month_abbreviation = {
            "Jan": 1,
            "Feb": 2,
            "Mar": 3,
            "Apr": 4,
            "May": 5,
            "Jun": 6,
            "Jul": 7,
            "Aug": 8,
            "Sep": 9,
            "Oct": 10,
            "Nov": 11,
            "Dec": 12,
        }

        vars_to_drop = ["booking_status", "date_of_reservation"]
        booking_map = {"Canceled": 1, "Not_Canceled": 0}

        preprocessor = NoVacancyDataProcessing(
            variable_rename=variable_rename,
            month_abbreviation=month_abbreviation,
            vars_to_drop=vars_to_drop,
            booking_map=booking_map,
        )

        X_silver, y_silver = preprocessor.fit_transform(valid_bronze_data)

        # Add back a column that should have been dropped
        invalid_silver = X_silver.copy()
        invalid_silver["booking_status"] = "Canceled"
        invalid_silver["is_cancellation"] = y_silver

        silver_results = validator.validate_silver_data(invalid_silver)
        assert silver_results["success"] is False
