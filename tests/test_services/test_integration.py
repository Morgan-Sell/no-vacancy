import pandas as pd
import pytest

from app.services.config_services import (
    BOOKING_MAP,
    MONTH_ABBREVIATION_MAP,
    VARIABLE_RENAME_MAP,
    VARS_TO_DROP,
    VARS_TO_IMPUTE,
    VARS_TO_OHE,
    SEARCH_SPACE
)
from app.services.data_management import DataManagement
from app.services.pipeline import NoVacancyPipeline
from app.services.predictor import make_prediction
from app.services.preprocessing import NoVacancyDataProcessing
from feature_engine.encoding import OneHotEncoder
from feature_engine.imputation import CategoricalImputer
from sklearn.ensemble import RandomForestClassifier


def test_end_to_end_pipeline(booking_data):
    # Step 1: Data Preprocessing
    processor = NoVacancyDataProcessing(
        variable_rename=VARIABLE_RENAME_MAP,
        month_abbreviation=MONTH_ABBREVIATION_MAP,
        vars_to_drop=VARS_TO_DROP,
        booking_map=BOOKING_MAP,
    )
    X_processed, y_processed = processor.fit_transform(
        booking_data.drop(columns=["booking status"]),
        booking_data["booking status"]
    )

    # Step 2: Pipeline training
    imputer = CategoricalImputer(imputation_method="frequent", variables=VARS_TO_IMPUTE)
    encoder = OneHotEncoder(variables=VARS_TO_OHE)
    estimator = RandomForestClassifier()

    pipeline = NoVacancyPipeline(imputer, encoder, estimator)
    pipeline.pipeline(SEARCH_SPACE)
    pipeline.fit(X_processed, y_processed)

    # Step 4: Prediction
    predictions = make_prediction(booking_data)

    # Assertions
    assert isinstance(predictions, pd.DataFrame)
    assert results.shape[0] == booking_data.shape[0]
    assert results.columns.tolist() == [
        "prediction",
        "probability_not_canceled",
        "probabilities_canceled",
    ]