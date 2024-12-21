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

    def pipeline(self, search_space):
        pipe = Pipeline(
            [
                ("cleaning_step", self.processor),
                ("imputation_step", self.imputer),
                ("encoding_step", self.encoder),
                ("model", self.estimator),
            ]
        )

        rscv = RandomizedSearchCV(
            pipe, search_space, cv=5, n_jobs=-1, return_train_score=False, verbose=3
        )
        return rscv
