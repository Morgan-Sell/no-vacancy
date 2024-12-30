from unittest.mock import MagicMock, patch

import pandas as pd
from feature_engine.encoding import OneHotEncoder
from feature_engine.imputation import CategoricalImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from app.services.config_services import (
    BOOKING_MAP,
    MONTH_ABBREVIATION_MAP,
    VARIABLE_RENAME_MAP,
    VARS_TO_DROP,
)
from app.services.preprocessing import NoVacancyDataProcessing


def test_pipeline_initization(sample_pipeline):
    assert isinstance(sample_pipeline.imputer, CategoricalImputer)
    assert isinstance(sample_pipeline.encoder, OneHotEncoder)
    assert isinstance(sample_pipeline.estimator, RandomForestClassifier)


def test_pipeline_structure(sample_pipeline, booking_data):
    """
    Ensure the pipeline is properly configured with the correct steps.
    """
    # Arrange
    search_space = {
        "model__n_estimators": list(range(1, 502, 50)),
        "model__max_features": ["log2", "sqrt"],
        "model__max_depth": [3, 5],
    }

    processor = NoVacancyDataProcessing(
        variable_rename=VARIABLE_RENAME_MAP,
        month_abbreviation=MONTH_ABBREVIATION_MAP,
        vars_to_drop=VARS_TO_DROP,
        booking_map=BOOKING_MAP,
    )
    X = booking_data.drop(columns=["booking status"])
    y = booking_data["booking status"]
    X_tr, y_tr = processor.fit_transform(X, y)

    # Action
    sample_pipeline.pipeline(search_space)
    sample_pipeline.fit(X_tr, y_tr)

    # Assert
    assert isinstance(sample_pipeline.rscv, RandomizedSearchCV)
    assert isinstance(sample_pipeline.pipe, Pipeline)
    assert "imputation_step" in sample_pipeline.pipe.named_steps
    assert "encoding_step" in sample_pipeline.pipe.named_steps
    assert "model" in sample_pipeline.pipe.named_steps


def test_pipeline_fit(sample_pipeline, booking_data):
    """
    Ensure the pipeline's training behavior is correct.
    """
    # Arrange
    search_space = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [3, 5, 7],
        "model__max_features": ["sqrt", "log2"],
    }

    processor = NoVacancyDataProcessing(
        variable_rename=VARIABLE_RENAME_MAP,
        month_abbreviation=MONTH_ABBREVIATION_MAP,
        vars_to_drop=VARS_TO_DROP,
        booking_map=BOOKING_MAP,
    )
    X = booking_data.drop(columns=["booking status"])
    y = booking_data["booking status"]
    X_tr, y_tr = processor.fit_transform(X, y)

    # Action
    sample_pipeline.pipeline(search_space)
    sample_pipeline.fit(X_tr, y_tr)

    # Assert
    # Verify RandomSearchCV is successfully fitted
    assert sample_pipeline.rscv is not None, "RandomizedSearchCV was not initialized."
    assert hasattr(
        sample_pipeline.rscv, "best_estimator_"
    ), "RandomizedSearchCV was not fitted."

    # Verify X and y have valid types
    assert isinstance(X_tr, pd.DataFrame)
    assert isinstance(y_tr, pd.Series)
