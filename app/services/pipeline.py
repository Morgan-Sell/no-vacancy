import logging
from typing import Union

from feature_engine.encoding import OneHotEncoder
from feature_engine.imputation import CategoricalImputer
from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from app.services.preprocessing import NoVacancyDataProcessing

_logger = logging.getLogger(__name__)


class NoVacancyPipeline:
    def __init__(
        self,
        processor: NoVacancyDataProcessing,
        imputer: CategoricalImputer,
        encoder: OneHotEncoder,
        clsfr: Union[BaseEstimator, XGBClassifier],
    ):
        self.processor = processor
        self.imputer = imputer
        self.encoder = encoder
        self.estimator = clsfr
        self.pipe = None  # Placeholder fo the constructed pipeline

    def pipeline(self, search_space):
        self.pipe = Pipeline(
            [
                ("cleaning_step", self.processor),
                ("imputation_step", self.imputer),
                ("encoding_step", self.encoder),
                ("model", self.estimator),
            ]
        )

        self.rscv = RandomizedSearchCV(
            estimator=self.pipe,
            param_distributions=search_space,
            cv=5,
            n_jobs=-1,
            return_train_score=False,
            verbose=3,
        )
        return self.rscv

    def fit(self, X, y):
        """Fit the pipeline using RandomizedSearchCV."""
        if not hasattr(self, "rscv"):
            raise AttributeError(
                "Pipeline not instantiated. Call `pipeline` method first."
            )
        self.rscv.fit(X, y)

    def predict_proba(self, X):
        """Make predictions using the best estimator from RandomizedSearchCV."""
        if not hasattr(self, "rscv"):
            raise AttributeError(
                "Pipeline not instantiated. Call `pipeline` method first."
            )
        return self.rscv.best_estimator_.predict_proba(X)
