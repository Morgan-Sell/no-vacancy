"""
Data schemas and expected values for NoVacancy validation.

This module defines the expected structure and values for data at each
pipeline stage (Bronze, Silver, Model Input). These schemas are used by
Great Expectations validators to catch data quality issues.

Design principle: Store expected values here as "data contracts" -
explicit agreements about what valid data looks like.
"""

# ============================================
# BRONZE LAYER: Raw CSV Schema
# ============================================
# Expected column order in bookings_raw.csv
BRONZE_COLUMNS = [
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
    "is_repeat_guest",
    "num_previous_cancellations",
    "num_previous_bookings_not_canceled",
    "average_price",
    "special_requests",
    "date_of_reservation",
    "booking_status",
]

# Expected categorical values (extracted from training data)
# WHY: OneHotEncoder expects fixed categories. New values = feature mismatch!
EXPECTED_CATEGORIES = {
    "booking_status": [
        "Canceled",
        "Not_Canceled",
    ],
    "type_of_meal": [
        "Meal Plan 1",
        "Meal Plan 2",
        "Meal Plan 3",
        "Not Selected",
    ],
    "room_type": [
        "Room_Type 1",
        "Room_Type 2",
        "Room_Type 3",
        "Room_Type 4",
        "Room_Type 5",
        "Room_Type 6",
        "Room_Type 7",
    ],
    "market_segment_type": [
        "Online",
        "Offline",
        "Corporate",
        "Aviation",
        "Complementary",
    ],
}

# Numerical constraints (business logic + data sanity)
# WHY: Detect corrupted data and data entry errors
NUMERICAL_BOUNDS = {
    "number_of_adults": {
        "min": 0,
        "max": 10,
        "mostly": 0.99,  # Allow 1% outliers
    },
    "number_of_children": {
        "min": 0,
        "max": 10,
        "mostly": 0.99,
    },
    "number_of_weekend_nights": {
        "min": 0,
        "max": 20,
        "mostly": 0.99,
    },
    "number_of_week_nights": {
        "min": 0,
        "max": 30,
        "mostly": 0.99,
    },
    "lead_time": {
        "min": 0,
        "max": 730,  # 2 years max advance booking
        "mostly": 0.99,
    },
    "average_price": {
        "min": 0,
        "max": 1000,  # Reasonable hotel price range
        "mostly": 0.99,
    },
    "special_requests": {
        "min": 0,
        "max": 10,
        "mostly": 0.99,
    },
}

# Critical columns that MUST NOT be null in Bronze layer
BRONZE_NON_NULL_COLUMNS = [
    "booking_id",  # Primary key
    "booking_status",  # Target variable
]


# ============================================
# SILVER LAYER: Transformed Data Schema
# ============================================
# Expected derived features created by NoVacancyDataProcessing
DERIVED_FEATURES = {
    "month_of_reservation": [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ],
    "day_of_week": [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ],
}

# Columns that should be REMOVED by preprocessing
# WHY: Verify transformer is working correctly
COLUMNS_TO_DROP = [
    "booking_status",  # Transformed to is_cancellation
    "date_of_reservation",  # Transformed to month/day features
]

# Target variable after transformation
SILVER_TARGET_VALUES = [0, 1]  # is_cancellation: 0=Not_Canceled, 1=Canceled


# ============================================
# MODEL INPUT LAYER: Final Feature Schema
# ============================================
# Note: Exact column list generated after OneHotEncoding
# We'll validate dynamically that all OHE columns are binary

# Expected feature prefixes after OneHotEncoding
# WHY: Validate OHE worked correctly
OHE_PREFIXES = [
    "is_type_of_meal_",
    "is_room_type_",
    "is_market_segment_type_",
    "is_month_of_reservation_",
    "is_day_of_week_",
]


# ============================================
# VALIDATION TOLERANCES
# ============================================
# How strict should validations be?
VALIDATION_CONFIG = {
    "mostly_threshold": 0.95,  # Allow 5% violations for categorical checks
    "outlier_threshold": 0.99,  # Allow 1% outliers for numerical checks
    "min_row_count": 1000,  # Minimum rows to consider data valid
    "max_row_count": 100000,  # Maximum rows (detect duplication)
}


# ============================================
# HELPER FUNCTIONS
# ============================================
def get_bronze_schema():
    """
    Returns complete Bronze layer schema for validation.

    Returns:
        dict: Schema definition with columns, categories, bounds
    """
    return {
        "columns": BRONZE_COLUMNS,
        "categories": EXPECTED_CATEGORIES,
        "numerical_bounds": NUMERICAL_BOUNDS,
        "non_null_columns": BRONZE_NON_NULL_COLUMNS,
    }


def get_silver_schema():
    """
    Returns complete Silver layer schema for validation.

    Returns:
        dict: Schema definition with derived features and dropped columns
    """
    return {
        "derived_features": DERIVED_FEATURES,
        "columns_to_drop": COLUMNS_TO_DROP,
        "target_values": SILVER_TARGET_VALUES,
    }


def get_validation_config():
    """
    Returns validation tolerance configuration.

    Returns:
        dict: Configuration for validation strictness
    """
    return VALIDATION_CONFIG
