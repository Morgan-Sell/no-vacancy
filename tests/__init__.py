import re

import pandas as pd
from services import BOOKING_MAP

NECESSARY_BINARY_VARIABLES = {
    # Add only fields defined in TrainData or ValidationTestData
    "type_of_meal": ["Meal Plan 1", "Meal Plan 2", "Meal Plan 3"],
    "room_type": [
        "Room_Type 1",
        "Room_Type 2",
        "Room_Type 3",
        "Room_Type 4",
        "Room_Type 5",
        "Room_Type 6",
        "Room_Type 7",
    ],
    "market_segment_type": ["Online", "Corporate", "Complementary", "Aviation"],
    "month_of_reservation": [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Oct",
        "Nov",
        "Dec",
    ],
    "day_of_week": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"],
}

BOOKING_DATA_VARS_TO_DROP = [
    "type_of_meal",
    "room_type",
    "market_segment_type",
    "month_of_reservation",
    "day_of_week",
    "date_of_reservation",
]

BOOKING_DATA_RENAME_MAP = {
    "repeated": "is_repeat_guest",
    "p_c": "num_previous_cancellations",
    "p_not_c": "num_previous_bookings_not_canceled",
    "booking_status": "is_cancellation",
}


# -- conftest.py helper functions --
def to_snake_case(name: str) -> str:
    # Replace hyphens (-) with underscores (_)
    name = name.replace("-", "_")

    # Preserve existing underscores and replace spaces or special characters with underscores
    name = re.sub(r"[^\w\s]", "", name)  # Remove special characters except underscores
    name = re.sub(r"(?<=[a-z])(?=[A-Z])", "_", name)  # Handle camelCase to snake_case
    name = re.sub(r"[\s]+", "_", name)  # Replace spaces with underscores
    return name.lower()


def convert_columns_to_snake_case(df: pd.DataFrame) -> pd.DataFrame:
    # Transform column names
    new_columns = [to_snake_case(col).replace("__", "_") for col in df.columns]
    df.columns = new_columns
    return df


def transform_booking_data_to_bronze_db_format(df):
    """
    Transform booking data to match the Bronze DB schema.
    """
    df = convert_columns_to_snake_case(df)
    df.rename(columns=BOOKING_DATA_RENAME_MAP, inplace=True)
    return df


def transform_booking_data_to_silver_db_format(df):
    """
    Transform booking data to match the Silver DB schema.

    """
    # Derive month and day columns for binary creation
    df["month_of_reservation"] = pd.to_datetime(df["date of reservation"]).dt.strftime(
        "%b"
    )
    df["day_of_week"] = pd.to_datetime(df["date of reservation"]).dt.strftime("%A")

    # Revise column names to match Silver DB schema
    df = convert_columns_to_snake_case(df)
    df.rename(columns=BOOKING_DATA_RENAME_MAP, inplace=True)

    # Create one-hot encoded columns for categorical variables
    for feature, values in NECESSARY_BINARY_VARIABLES.items():
        for val in values:
            col_name = f"is_{feature.lower()}_{val.lower()}".replace(" ", "_")
            df[col_name] = (df[feature] == val).astype(int)

    df["is_cancellation"] = df["is_cancellation"].map(BOOKING_MAP)

    # Drop original categorical columns
    df.drop(columns=BOOKING_DATA_VARS_TO_DROP, inplace=True)

    return df


def get_db_model_column_names(model):
    return set(col.name for col in model.__table__.columns)
