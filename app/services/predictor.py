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
    raise error_type(f"{message}: {exception}") #from exception


def make_prediction(test_data: pd.DataFrame):

    try:
        if not isinstance(test_data, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        
        if test_data.empty:
            raise ValueError("Input data is empty. Cannot make predictions on an empty DataFrame.")
        
        # Load pipeline
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


            
