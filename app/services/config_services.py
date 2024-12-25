# -- Data Processing --
VARIABLE_RENAME_MAP = {
    "repeated": "is_repeat_guest",
    "pc": "num_previous_cancellations",
    "pnot_c": "num_previous_bookings_not_canceled",
}

MONTH_ABBREVIATION_MAP = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec",
}

BOOKING_MAP = {
    "Not_Canceled": 0,
    "Canceled": 1,
}

VARS_TO_DROP = ["Booking_ID", "date of reservation"]

# -- Feature Engineering --
VARS_TO_IMPUTE = ["month_of_reservation", "day_of_week"]

VARS_TO_OHE = [
    "type_of_meal",
    "room_type",
    "market_segment_type",
    "month_of_reservation",
    "day_of_week",
]

# -- Model Training --
SEARCH_SPACE = {
    "model__n_estimators": list(range(1, 502, 50)),
    "model__max_features": ["log2 ", "sqrt "],
    "model__max_depth": list(range(1, 32, 5)),
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__bootstrap": [True, False],
}

TRAIN_RATIO = 0.7

TARGET_VARIABLE = "booking status"

# TODO: Update paths
DATA_PATHS = {
    "raw_data": "data/raw/no_vacancy.csv",
    "processed_data": "data/processed/processed_no_vacancy.csv",
    "model_save_path": "data/models/xgboost_model.pkl",
}
