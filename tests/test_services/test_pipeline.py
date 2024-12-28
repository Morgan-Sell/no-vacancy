from unittest.mock import MagicMock, patch

import pandas as pd
from feature_engine.encoding import OneHotEncoder
from feature_engine.imputation import CategoricalImputer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from app.services.config_services import BOOKING_MAP, MONTH_ABBREVIATION_MAP, VARIABLE_RENAME_MAP, VARS_TO_DROP
from app.services.preprocessing import NoVacancyDataProcessing


def test_pipeline_initization(sample_pipeline):
    assert isinstance(sample_pipeline.imputer, CategoricalImputer)
    assert isinstance(sample_pipeline.encoder, OneHotEncoder)
    assert isinstance(sample_pipeline.estimator, RandomForestClassifier)


def test_pipeline_structure(sample_pipeline, booking_data):
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


@patch("sklearn.model_selection.RandomizedSearchCV.fit")
def test_pipeline_fit(mock_fit, sample_pipeline, booking_data):
    # Arrange
    search_space = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [3, 5],
    }
    sample_pipeline.pipeline(search_space)

    # Action
    sample_pipeline.fit(
        booking_data.drop(columns=["booking status"]), booking_data["booking status"]
    )

    # Assert
    # Verify that fit() method of RandomizedSearchCV was only called once
    mock_fit.assert_called_once()

    # Verify that X and y are passed as arguments
    fit_args = mock_fit.call_args[0]
    assert isinstance(fit_args[0], pd.DataFrame)
    assert isinstance(fit_args[1], pd.Series)
