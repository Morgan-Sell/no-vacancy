# ------------------------------------------
# Global Variables for the services module
# ------------------------------------------

# -- Data Processing --
DEPENDENT_VAR_NAME = "booking_status"


VARIABLE_RENAME_MAP = {
    "repeated": "is_repeat_guest",
    "p_c": "num_previous_cancellations",
    "p_not_c": "num_previous_bookings_not_canceled",
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

VARS_TO_DROP = [
    # "booking_id",
    "date_of_reservation"
]

# -- Data Preprocessing (preprocessing.py) --
VARS_TO_IMPUTE = ["month_of_reservation", "day_of_week"]

VARS_TO_OHE = [
    "type_of_meal",
    "room_type",
    "market_segment_type",
    "month_of_reservation",
    "day_of_week",
]

# -- Pipeline (pipeline.py) --
PRIMARY_KEY = "booking_id"


RSCV_PARAMS = {
    "n_iter": 20,  # TODO: Update to 50 after debugging
    "scoring": "roc_auc",
    "n_jobs": 1,
    "cv": 3,  # TODO: Update to 5 after debugging
    "verbose": 1,
    "return_train_score": False,
}


# -- Model Training (trainer.py) --
# model__ not required b/c RCSV is applied before the Pipeline
SEARCH_SPACE = {
    "n_estimators": list(range(1, 502, 50)),
    "max_features": ["log2", "sqrt"],
    "max_depth": list(range(1, 32, 5)),
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "bootstrap": [True, False],
}

TRAIN_RATIO = 0.7

IMPUTATION_METHOD = "frequent"

RAW_TARGET_VARIABLE = "booking_status"


DATA_PATHS = {
    "model_save_path": "models/no_vacancy_pipeline.pkl",
}

# -- Model Inference (predictor.py) --
SILVER_DB_TARGET_VARIABLE = "is_cancellation"

# -- MLflow Tracking --
MLFLOW_TRACKING_URI = "http://mlfow:50000"
MLFLOW_PROCESSOR_JOBLIB = "processor.joblib"
MLFLOW_PROCESSOR_PATH = f"processor/{MLFLOW_PROCESSOR_JOBLIB}"
MLFLOW_EXPERIMENT_NAME = "NoVacancyModelTraining"
