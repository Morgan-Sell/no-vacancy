"""
Data validators for NoVacancy ML pipeline using Great Expectations.

This module implements validation at three critical checkpoints:
1. Bronze Layer: Validate raw CSV data quality
2. Silver Layer: Verify preprocessing transformations
3. Model Input: Detect schema drift before inference

Design Philosophy:
- Fail fast: Stop pipeline immediately if data quality issues detected
- Fail loudly: Log clear error messages for debugging
- Focus on high-impact validations (12 total, not 50+)
"""

from pathlib import Path
from typing import Any, Dict

import great_expectations as gx
import pandas as pd
from config import get_logger
from great_expectations.core import ExpectationSuite
from validations.schemas import (
    BRONZE_COLUMNS,
    BRONZE_NON_NULL_COLUMNS,
    DERIVED_FEATURES,
    EXPECTED_CATEGORIES,
    NUMERICAL_BOUNDS,
    OHE_PREFIXES,
    SILVER_TARGET_VALUES,
    VALIDATION_CONFIG,
)

logger = get_logger(logger_name=__name__)


class DataQualityError(Exception):
    """Raised when data validation fails."""

    pass


class NoVacancyDataValidator:
    """
    Validates data quality at critical pipeline checkpoints.

    Validation Strategy:
    - Bronze: Catch corrupted/malformed raw data (8 validations)
    - Silver: Verify transformations executed correctly (3 validations)
    - Model Input: Detect schema drift before inference (1+ validations)
    """

    def __init__(self):
        """Initialize Great Expecations context."""
        validation_root = Path(__file__).parent
        self.context = gx.get_context(
            mode="file", project_root_dir=str(validation_root)
        )
        logger.info("✅ NoVacancyDataValidator initialized")

    # ============================================
    # BRONZE LAYER VALIDATION
    # ============================================

    def validate_bronze_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate raw CSV data after import (Bronze layer).

        Checks performed:
        1. Schema integrity - correct columns present in order
        2. Primary key validity - unique, non-null booking_ids
        3. Categorical domains - no unexpected categories (prevents OHE issues)
        4. Business constraints - reasonable value ranges

        Args:
            df: Raw DataFrame from bookings_raw.csv

        Returns:
            dict: {
                "success": bool,
                "statistics": {...},
                "failed_expectations": [...]
            }

        Raises:
            DataQualityError: If validation fails and strict mode enabled
        """
        logger.info("Starting Bronze layer validation...")

        # Get validator
        validator = self._get_validator(df, "bronze_bookings", "bronze_suite")

        # === VALIDATION 1: Schema Structure ===
        # Ensures CSV matches expected structure
        validator.expect_table_columns_to_match_ordered_list(
            column_list=BRONZE_COLUMNS,
            meta={
                "validation_layer": "bronze",
                "criticality": "critical",
                "description": "Column order much match training data",
            },
        )

        # === VALIDATION 2: Row Count Sanity ===
        # Detects truncated files or accidental duplication
        validator.expect_table_row_count_to_be_between(
            min_value=VALIDATION_CONFIG["min_row_count"],
            max_value=VALIDATION_CONFIG["max_row_count"],
            meta={
                "validation_layer": "bronze",
                "criticality": "high",
                "description": "File size should be reasonable",
            },
        )

        # === VALIDATION 3: Primary Key Uniqueness ===
        # Duplicate IDs corrupt predictions in Gold DB
        validator.expect_column_values_to_be_unique(
            column="booking_id",
            meta={
                "validation_layer": "bronze",
                "criticality": "critical",
                "desrciption": "booking_id must be unique",
            },
        )

        # === VALIDATION 4: Critical Nulls ===
        # Primary key and target cannot be null
        for col in BRONZE_NON_NULL_COLUMNS:
            validator.expect_column_values_to_not_be_null(
                column=col,
                meta={
                    "validation_layer": "bronze",
                    "criticality": "critical",
                    "description": f"{col} required for processing",
                },
            )

        # === VALIDATION 5-8: Categorical Domain Validation ===
        # Prevents OneHotEncoder from creating mismatched features
        for col, expected_values in EXPECTED_CATEGORIES.items():
            validator.expect_column_values_to_be_in_set(
                column=col,
                value_set=expected_values,
                mostly=VALIDATION_CONFIG["mostly_threshold"],
                meta={
                    "validation_layer": "bronze",
                    "criticality": "critical",
                    "description": f"Unexpected {col} values break OHE",
                },
            )

        # === VALIDATION 9-15: Numerical Bounds ====
        # Detect data corruption and entry errors
        for col, bounds in NUMERICAL_BOUNDS.items():
            validator.expect_column_values_to_be_between(
                column=col,
                min_value=bounds["min"],
                max_value=bounds["max"],
                mostly=bounds["mostly"],
                meta={
                    "validation_layer": "bronze",
                    "criticality": "medium",
                    "description": f"{col} should be within business logic bounds",
                },
            )

        # Save suite and run validation
        self.context.suites.add_or_update(validator.expectation_suite)
        results = validator.validate()

        return self._format_results(results, "Bronze")

    # ============================================
    # SILVER LAYER VALIDATION
    # ============================================

    def validate_silver_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate transformed features (Silver layer).

        Checks performed:
        1. Derived features created correclty (month, day_of_week)
        2. Target transformed to binary (0/1)
        3. Expected columns dropped by preprocessor

        Args:
            df: Transformed DataFrame after NoVacancyDataProcessing

        Returns:
            dict: Validation results

        Raises:
            DataQualityError: If transformation validation fails
        """
        logger.info("Starting Silver layer validation...")

        validator = self._get_validator(df, "silver_processed", "silver_suite")

        # === VALIDATION 1: Month Extraction ===
        # Validates data parsing worked correctly
        validator.expect_column_values_to_be_in_set(
            column="month_of_reservation",
            value_set=DERIVED_FEATURES["month_of_reservation"],
            meta={
                "validation_layer": "silver",
                "criticality": "high",
                "description": "Month extraction from date much be valid",
            },
        )

        # === VALIDATION 2: Day of Week Extraction ===
        # Validates data parsing worked correctly
        validator.expect_column_values_to_be_in_set(
            column="day_of_week",
            value_set=DERIVED_FEATURES["day_of_week"],
            meta={
                "validation_layer": "silver",
                "criticality": "high",
                "description": "Day extraction from date must be valid",
            },
        )

        # === VALIDATION 3: Target Transformation ===
        # Target variable should be binary
        validator.expect_column_values_to_be_in_set(
            column="is_cancellation",
            value_set=SILVER_TARGET_VALUES,
            meta={
                "validation_layer": "silver",
                "criticaliy": "critical",
                "description": "Target must be binary (0/1)",
            },
        )

        # Save and validate
        self.context.suites.add_or_update(validator.expectation_suite)
        results = validator.validate()

        return self._format_results(results, "Silver")

    # ============================================
    # MODEL INPUT VALIDATION
    # ============================================

    def validate_model_input(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data before model inference (Model Input layer).

        Checks performed:
        1. All OneHotEncoded columns are binary (0/1)
        2. Expected feature prefixes exist

        Args:
            df: Final feature matrix ready for model.predict()

        Returns:
            dict: Validation results

        Raises:
            DataQualityError: If schema drift detected
        """
        logger.info("Starting Model Input validation...")

        validator = self._get_validator(df, "model_input", "model_input_suite")

        # === VALIDATION 1: OHE Binary Check ===
        # OneHotEncoder should only create 0/1 values
        ohe_columns = [col for col in df.columns if col.startswith("is_")]

        if not ohe_columns:
            logger.warning("⚠️  No OneHotEncoded columns found - is this expected?")

        for col in ohe_columns:
            validator.expect_column_values_to_be_in_set(
                column=col,
                value_set=[0, 1],
                meta={
                    "validation_layer": "model_input",
                    "criticality": "critical",
                    "description": "OHE columns must be binary",
                },
            )

        # === VALIDATION 2: Expected Prefixes Exist ===
        # Validates all categorical features were encoded
        for prefix in OHE_PREFIXES:
            prefix_columns = [col for col in df.columns if col.startswith(prefix)]

            if not prefix_columns:
                logger.warning(f"⚠️  No columns found with prefix '{prefix}'")

        # Save and validate
        self.context.suites.add_or_update(validator.expectation_suite)
        results = validator.validate()

        return self._format_results(results, "Model Input")

    # ============================================
    # HELPER METHODS
    # ============================================

    def _get_validator(
        self, df: pd.DataFrame, asset_name: str, suite_name: str
    ) -> gx.core.ExpectationValidationResult:
        """
        Create Great Expecations validator for dataframe.

        Args:
            df: DataFrame to validate
            asset_name: Data asset name (e.g., bronze_bookings, silver_processed)
            suite_name: Expectation suite name

        Returns:
            GE Validator instance
        """

        # Get datasource and asset
        datasource = self.context.data_sources.get("pandas_datasource")
        data_asset = datasource.get_asset(asset_name)

        # Create batch from dataframe
        batch_request = data_asset.build_batch_request(options={"dataframe": df})

        # Get or create expectation suite
        try:
            suite = self.context.suites.get(suite_name)
        except Exception:
            suite = ExpectationSuite(name=suite_name)
            self.context.suites.add_or_update(suite)

        # Return validator
        return self.context.get_validator(
            batch_request=batch_request, expectation_suite=suite
        )

    def _format_results(self, results, layer_name: str) -> Dict[str, Any]:
        """
        Format validation results with logging.

        Args:
            results: GE validation results
            layer_name: Name of validation layer (Bronze/Silver/Model Input)

        Returns:
            dict: Formatted results with success flag and statistics
        """
        if results.success:
            logger.info(f"✅ {layer_name} validation PASSED")

        else:
            logger.error(f"❌ {layer_name} validation FAILED")

            # Log each failed expectation
            for result in results.results:
                if not result.success:
                    expectation_type = result.expectation_config.type
                    logger.error(f"  ❌ Failed: {expectation_type}")

                    # Log additional context if available
                    if hasattr(result.expectation_config, "kwargs"):
                        kwargs = result.expectation_config.kwargs
                        if "column" in kwargs:
                            logger.error(f"     Column: {kwargs['column']}")

        return {
            "success": results.success,
            "statistics": results.statistics,
            "failed_expectations": [r for r in results.results if not r.success],
            "results": results,
        }

    def validate_and_raise(
        self, df: pd.DataFrame, layer: str = "bronze"
    ) -> Dict[str, Any]:
        """
        Conveneince method: validate and raise excpetion if fails.

        Args:
            df: Dataframe to validate
            layer: Which layer to validate (bronze/silver/model_input)

        Returns:
            dict: Validation results

        Raises:
            DataQualityError: If validation fails

        Example:
            >>> validator = NoVacancyValidator()
            >>> validator.validate_and_raise(df, layer="bronze")
        """
        if layer == "bronze":
            results = self.validate_bronze_data(df)
        elif layer == "silver":
            results = self.validate_silver_data(df)
        elif layer == "model_input":
            results = self.validate_model_input(df)
        else:
            raise ValueError(f"Unknown layer: {layer}")

        if not results["success"]:
            raise DataQualityError(
                f"{layer.capitalize()} layer validation failed! "
                f"Check logs for details."
            )

        return results
