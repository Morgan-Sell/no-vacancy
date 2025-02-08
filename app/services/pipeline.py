import logging
from typing import Union

import pandas as pd
from feature_engine.encoding import OneHotEncoder
from feature_engine.imputation import CategoricalImputer
from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from services import RSCV_PARAMS

_logger = logging.getLogger(__name__)


class NoVacancyPipeline:
    """
    A custom pipeline for training and predicting on the NoVacancy dataset.

    NoVacancyDataProcessing is omitted because it transforms the target variable,
    which causes issues with the pipeline.
    """

    def __init__(
        self,
        imputer: CategoricalImputer,
        encoder: OneHotEncoder,
        clsfr: Union[BaseEstimator, XGBClassifier],
    ):
        self.imputer = imputer
        self.encoder = encoder
        self.estimator = clsfr
        self.pipe = None  # Placeholder fo the constructed pipeline
        self.rscv = None  # Placeholder for the RandomizedSearchCV object

    def pipeline(self, search_space):
        self.pipe = Pipeline(
            [
                ("imputation_step", self.imputer),
                ("encoding_step", self.encoder),
                ("model", self.estimator),
            ]
        )

        self.rscv = RandomizedSearchCV(
            estimator=self.pipe,
            param_distributions=search_space,
            **RSCV_PARAMS,
        )

    def fit(self, X, y):
        """Fit the pipeline using RandomizedSearchCV."""
        if self.rscv is None:
            raise AttributeError(
                "Pipeline not instantiated. Call `pipeline` method first."
            )
        self.rscv.fit(X, y)


    def transform(self, X):
        """Apply imputation and encoding transformations to input data."""
        if self.rscv is None:
            raise AttributeError(
                "Pipeline not instantiated. Call `pipeline` method first."
            )
        
        print("\n[DEBUG] Transforming input data...")

        # Apply imputation
        imputer = self.rscv.best_estimator_.named_steps["imputation_step"]
        print("\n[DEBUG] Before imputation columns:", X.columns)

        X_imputed = imputer.transform(X)
        print("\n[DEBUG] After imputation columns:", X_imputed.columns)

        # Apply encoding
        encoder = self.rscv.best_estimator_.named_steps["encoding_step"]
        print("\n[DEBUG] Before encoding columns:", X_imputed.columns)

        X_encoded = encoder.transform(X_imputed)
        print("\n[DEBUG] After encoding columns:", X_encoded.columns)

        return X_encoded
    

    def predict_proba(self, X):
        """Make predictions using the best estimator from RandomizedSearchCV."""
        if not hasattr(self, "rscv"):
            raise AttributeError(
                "Pipeline not instantiated. Call `pipeline` method first."
            )
        
        X_transformed = self.transform(X)
        return self.rscv.best_estimator_.predict_proba(X_transformed)

    def predict(self, X):
        """Make class predictions using the best estimator from RandomizedSearchCV."""
        if not hasattr(self, "rscv"):
            raise AttributeError(
                "Pipeline not instantiated. Call `pipeline` method first."
            )
        
        X_transformed = self.transform(X)
        return self.rscv.best_estimator_.predict(X_transformed)
