import pandas as pd
import pytest
from feature_engine.encoding import OneHotEncoder
from feature_engine.imputation import CategoricalImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from app.services.config_services import (
    BOOKING_MAP,
    DATA_PATHS,
    MONTH_ABBREVIATION_MAP,
    SEARCH_SPACE,
    VARIABLE_RENAME_MAP,
    VARS_TO_DROP,
    VARS_TO_IMPUTE,
    VARS_TO_OHE,
)
from app.services.data_management import DataManagement
from app.services.pipeline import NoVacancyPipeline
from app.services.predictor import make_prediction
from app.services.preprocessing import NoVacancyDataProcessing


def test_end_to_end_pipeline(booking_data, dm, temp_pipeline_path):
    # Step 1: Split data into train and test
    X = booking_data.drop(columns=["booking status"])
    y = booking_data["booking status"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=33
    )

    # Step 2: Data Preprocessing
    processor = NoVacancyDataProcessing(
        variable_rename=VARIABLE_RENAME_MAP,
        month_abbreviation=MONTH_ABBREVIATION_MAP,
        vars_to_drop=VARS_TO_DROP,
        booking_map=BOOKING_MAP,
    )
    X_prcsd, y_prcsd = processor.fit_transform(X_train, y_train)

    # Assertions: Data Preprocessing
    assert isinstance(X_prcsd, pd.DataFrame), "X_prcsd is not a DataFrame."
    assert isinstance(y_prcsd, pd.Series), "y_prcsd is not a Series."
    assert (
        X_prcsd.shape[0] == y_prcsd.shape[0]
    ), "X_prcsd and y_prcsd row counts do not equal "

    # Step 3: Train pipeline
    imputer = CategoricalImputer(imputation_method="frequent", variables=VARS_TO_IMPUTE)
    encoder = OneHotEncoder(variables=VARS_TO_OHE)
    estimator = RandomForestClassifier()

    pipeline = NoVacancyPipeline(imputer, encoder, estimator)
    pipeline.pipeline(SEARCH_SPACE)
    pipeline.fit(X_prcsd, y_prcsd)

    # Asserts: Pipeline training
    assert hasattr(pipeline, "rscv"), "Pipeline does not have rscv attribute."
    assert pipeline.rscv.best_estimator_ is not None, "Best estimator is None."

    # Step 4: Save pipeline using the dm pytest fixture
    dm.save_pipeline(pipeline, processor)

    # Assertions: Pipeline saving
    assert (
        dm.pipeline_path == temp_pipeline_path
    ), "Pipeline path does not match temp path."

    # Step 5: Load pipeline & processor
    loaded_pipeline, loaded_processor = dm.load_pipeline()

    # Assertions: Pipeline & Processor loading
    assert loaded_pipeline is not None, "Loaded pipeline is None."
    assert hasattr(
        loaded_pipeline, "predict"
    ), "Loaded pipeline missing predict method."
    assert loaded_processor is not None, "Loaded processor is None."
    assert hasattr(
        loaded_processor, "transform"
    ), "Loaded processor missing transform method."

    # Step 5: Preprocess and align test data
    X_test_prcsd, y_test_prcsd = processor.transform(X_test, y_test)
    expected_columns = loaded_pipeline.rscv.best_estimator_.named_steps[
        "encoding_step"
    ].get_feature_names_out()
    X_test_prcsd = X_test_prcsd.reindex(columns=expected_columns, fill_value=0)
    print("Test data columns before predictions:", X_test_prcsd.columns)

    # Step 6: Prediction
    predictions = make_prediction(X_test_prcsd, dm)

    # Assertions: Prediction
    assert isinstance(predictions, pd.DataFrame), "Predictions are not a DataFrame."
    assert (
        predictions.shape[0] == X_test_prcsd.shape[0]
    ), "Prediction row count does not match input data."
    assert predictions.columns.tolist() == [
        "prediction",
        "probability_not_canceled",
        "probabilities_canceled",
    ], "Prediction columns do not match expected output."

    # Validation prediction results
    assert (
        predictions["prediction"].isin([0, 1]).all()
    ), "Predictions contain invalid values."
    assert (
        predictions["probability_not_canceled"].between(0, 1).all()
    ), "Probability values are out of bounds."
    assert (
        predictions["probabilities_canceled"].between(0, 1).all()
    ), "Probability values are out of bounds."
