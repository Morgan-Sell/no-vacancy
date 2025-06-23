import logging
from typing import List, Union

from feature_engine.encoding import OneHotEncoder
from feature_engine.imputation import CategoricalImputer
from services import RSCV_PARAMS
from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

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
        vars_to_drop: List[str],
    ):
        self.imputer = imputer
        self.encoder = encoder
        self.estimator = clsfr
        self.vars_to_drop = vars_to_drop
        self.pipe = None  # Placeholder fo the constructed pipeline
        self.rscv = None  # Placeholder for the RandomizedSearchCV object
        self.X_initial = None  # Placeholder for the initial feature set
        self.X_encoded = None  # Placeholder for the final transformed feature set

    def fit(self, X, y, search_space):
        """Fit imputer and encoder separately, then train the model using transformed data."""
        # Drop unnecessary columns
        X.drop(columns=self.vars_to_drop, inplace=True, errors="ignore")

        # Store the initial features for logging
        self.X_initial = X.copy()

        # Fit imputer and transform X
        self.imputer.fit(X)
        X_imputed = self.imputer.transform(X)

        # Fit encoder and transform X
        self.encoder.fit(X_imputed)
        X_encoded = self.encoder.transform(X_imputed)

        # Store the encoded features for later use
        self.X_encoded = X_encoded

        # Fit the model using transformed X
        self.rscv = RandomizedSearchCV(
            estimator=self.estimator,
            param_distributions=search_space,
            **RSCV_PARAMS,
        )
        self.rscv.fit(X_encoded, y)

        # Create the final pipeline with the best estimator
        self.pipe = Pipeline(
            [
                ("imputation_step", self.imputer),
                ("encoding_step", self.encoder),
                ("model", self.rscv.best_estimator_),
            ]
        )

        return self

    def predict(self, X):
        """Predict class using the final pipeline (with imputer and encoder)."""
        if self.pipe is None:
            raise AttributeError("Pipeline is not trained. Call 'fit' method first.")

        return self.pipe.predict(X)

    def predict_proba(self, X):
        """Predict probabilities using the final pipeline (with imputer and encoder)."""
        if self.pipe is None:
            raise AttributeError("Pipeline is not trained. Call 'fit' method first.")
        return self.pipe.predict_proba(X)

    def get_logged_params(self):
        """Get selected parameters to be logged."""
        if self.rscv is None:
            raise AttributeError(
                "Model is not trained. Call 'fit' before retrieving parameters."
            )

        # Collect information about the features used in the model
        feature_columns = (
            self.X_encoded.columns.tolist() if self.X_encoded is not None else []
        )
        ohe_vars_created = [col for col in feature_columns if col.startswith("is_")]

        return {
            "num_initial_vars": (
                len(self.X_initial.columns) if self.X_initial is not None else 0
            ),
            "total_num_features_for_training": len(feature_columns),
            "imputer_type": self.imputer.__class__.__name__,
            "imputation_method": getattr(self.imputer, "imputation_method", None),
            "num_imputer_vars": getattr(self.imputer, "n_features_in_", None),
            "imputer_vars": getattr(self.imputer, "variables", None),
            "encoder_type": self.encoder.__class__.__name__,
            "num_encoder_vars": getattr(self.encoder, "n_features_in_", None),
            "encoder_vars": getattr(self.encoder, "variables", None),
            "num_ohe_vars_created": len(ohe_vars_created),
            "model": self.estimator.__class__.__name__,
            **{f"model_param_{k}": v for k, v in self.rscv.best_params_.items()},
            "best_model_val_score": round(self.rscv.best_score_, 5),
        }

    def get_full_pipeline(self):
        """
        Returns the full trained pipeline (imputer → encoder → model).
        Useful for MLflow logging or inference.
        """
        if self.pipe is None:
            raise AttributeError("Pipeline is not trained. Call 'fit' method first.")
        return self.pipe
