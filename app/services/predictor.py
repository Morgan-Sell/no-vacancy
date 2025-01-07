import logging

import numpy as np
import pandas as pd
import pytest

from app.config import __model_version__
from app.services.config_services import (
    BOOKING_MAP,
    MONTH_ABBREVIATION_MAP,
    VARIABLE_RENAME_MAP,
    VARS_TO_DROP,
)
from app.services.data_management import DataManagement
from app.services.preprocessing import NoVacancyDataProcessing

logger = logging.getLogger(__name__)


def handle_error(error_type, message, exception):
    logger.error(f"{message}: {exception}")
    raise error_type(f"{message}: {exception}")  # from exception


def make_prediction(test_data: pd.DataFrame, dm: DataManagement = None):

    try:
        if not isinstance(test_data, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        if test_data.empty:
            raise ValueError(
                "Input data is empty. Cannot make predictions on an empty DataFrame."
            )

        # Load pipeline
        if dm is None:
            dm = DataManagement()
        pipeline = dm.load_pipeline()

        # Preprocess input data
        X_test = test_data.copy()

        processor = NoVacancyDataProcessing(
            variable_rename=VARIABLE_RENAME_MAP,
            month_abbreviation=MONTH_ABBREVIATION_MAP,
            vars_to_drop=VARS_TO_DROP,
            booking_map=BOOKING_MAP,
        )
        X_test_prcsd, _ = processor.transform(X_test)

        # Extract feature names from the imputer step
        imputer = pipeline.rscv.best_estimator_.named_steps["imputation_step"]
        encoder = pipeline.rscv.best_estimator_.named_steps["encoding_step"]

        # Explicitly set metadata if it's missing; otherwise, the pipeline will fail b/c
        # X_train_prcsd and X_train_prcsd have different "columns"
        if not hasattr(imputer, "n_features_in_"):
            imputer.n_features_in_ = X_test_prcsd.shape[1]

        if not hasattr(encoder, "n_features_in_"):
            encoder.n_features_in_ = X_test_prcsd.shape[1]

        # Reconcile training and test data shapes
        # OHE columns in test data may not exist in training data
        # This ensures that the test data has the same columns as the training data
        expected_columns = pipeline.rscv.best_estimator_.named_steps[
            "encoding_step"
        ].get_feature_names_out()
        X_test_prcsd = X_test_prcsd.reindex(columns=expected_columns, fill_value=0)

        # TODO: Omit columns that exist in the test dataset, but not the training dataset

        # TODO: Add a check to ensure that the test data has the same columns as the training data
        print("X_train_prcsd columns: ", X_train_prcsd.columns)
        print("X_train_prcsd shape: ", X_train_prcsd.shape)       
        print("X_test_prcsd columns: ", X_test_prcsd.columns)
        print("X_test_prcsd shape: ", X_test_prcsd.shape)

        # Generate the predictions using the pipeline
        predictions = pipeline.predict(X_test_prcsd)
        probabilities = pipeline.predict_proba(X_test_prcsd)

        probabilities = np.array(probabilities)

        # Format predictions into a dataframe
        results = pd.DataFrame(
            {
                "prediction": predictions,
                "probability_not_canceled": probabilities[:, 0],
                "probabilities_canceled": probabilities[:, 1],
            }
        )

        return results

    except ValueError as e:
        handle_error(ValueError, "❌ Invalid input", e)
    except FileNotFoundError as e:
        handle_error(FileNotFoundError, "❌ No pipeline found", e)
    except Exception as e:
        handle_error(RuntimeError, "❌ Prediction failed", e)
