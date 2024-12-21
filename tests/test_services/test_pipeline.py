from unittest.mock import MagicMock, patch

from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

from app.services.preprocessing import NoVacancyDataProcessing
from feature_engine.encoding import OneHotEncoder
from feature_engine.imputation import CategoricalImputer

from sklearn.pipeline import Pipeline


def test_pipeline_initization(sample_pipeline):
    assert isinstance(sample_pipeline.processor, NoVacancyDataProcessing)
    assert isinstance(sample_pipeline.imputer, CategoricalImputer)
    assert isinstance(sample_pipeline.encoder, OneHotEncoder)
    assert isinstance(sample_pipeline.estimator, XGBClassifier)


def test_pipeline_structure(sample_pipeline):
    # Arrange
    search_space = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [3, 4, 5],
        "model__learning_rate": [0.1, 0.01, 0.001],
    }

    # Action
    pipeline = sample_pipeline.pipeline(search_space)

    # Assert
    assert isinstance(pipeline, RandomizedSearchCV)
    assert isinstance(pipeline.estimator, Pipeline)
    assert "cleaning_step" in pipeline.estimator.named_steps
    assert "imputation_step" in pipeline.estimator.named_steps
    assert "encoding_step" in pipeline.estimator.named_steps
    assert "model" in pipeline.estimator.named_steps


    