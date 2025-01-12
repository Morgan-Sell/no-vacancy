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
        pipeline, processor = dm.load_pipeline()

        print("\nmake_predictions test_data (raw input): ", test_data.columns)

        # Process test data using loaded processor
        X_test_prcsd, _ = processor.transform(test_data)

        print("\nmake_predictions X_test_prcsd columns after processing: ", X_test_prcsd.columns)

        # Extract feature names from the imputer step
        imputer = pipeline.rscv.best_estimator_.named_steps["imputation_step"]
        encoder = pipeline.rscv.best_estimator_.named_steps["encoding_step"]

        # Explicitly set metadata if it's missing; otherwise, the pipeline will fail b/c
        # X_train_prcsd and X_train_prcsd have different "columns"
        if not hasattr(imputer, "n_features_in_"):
            imputer.n_features_in_ = X_test_prcsd.shape[1]

        if not hasattr(encoder, "n_features_in_"):
            encoder.n_features_in_ = X_test_prcsd.shape[1]

        # Ensure column consistency with the pipeline's expected columns
        # OHE columns in training data may not exist in test data
        expected_columns = pipeline.rscv.best_estimator_.named_steps[
            "encoding_step"
        ].get_feature_names_out()


        # Align test data with the expected columns
        X_test_prcsd = X_test_prcsd.reindex(columns=expected_columns, fill_value=0)

        # Explicitly update medatadata ('n_features_in_') in the transformers
        imputer = pipeline.rscv.best_estimator_.named_steps["imputation_step"]
        encoder = pipeline.rscv.best_estimator_.named_steps["encoding_step"]

        if hasattr(imputer, "n_features_in_"):
            imputer.n_features_in_ = X_test_prcsd.shape[1]
        if hasattr(encoder, "n_features_in_"):
            encoder.n_features_in_ = X_test_prcsd.shape[1]

        # Validate metadata consistency
        assert (
            imputer.n_features_in_ == X_test_prcsd.shape[1]
        ), "❌ Imputer metadata mismatch: n_features_in_ does not match the number of test dataset columns."
        assert (
            encoder.n_features_in_ == X_test_prcsd.shape[1]
        ), "❌ Encoder metadata mismatch: n_features_in_ does not match the number of test dataset columns."
        # TODO: Omit columns that exist in the test dataset, but not the training dataset

        print("make_predictions X_test_prcsd columns after reindexing: ", X_test_prcsd.columns)
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
