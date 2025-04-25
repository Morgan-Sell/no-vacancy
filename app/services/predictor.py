import logging

import numpy as np
import pandas as pd
from app.config import __model_version__
from app.services import (
    BOOKING_MAP,
    DEPENDENT_VAR_NAME,
    MONTH_ABBREVIATION_MAP,
    VARIABLE_RENAME_MAP,
    VARS_TO_DROP,
)
from app.services.pipeline_management import PipelineManagement
from app.services.preprocessing import NoVacancyDataProcessing

logger = logging.getLogger(__name__)


def handle_error(error_type, message, exception):
    logger.error(f"{message}: {exception}")
    raise error_type(f"{message}: {exception}")  # from exception


def make_prediction(test_data: pd.DataFrame, pm: PipelineManagement = None):
    try:
        if not isinstance(test_data, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        if test_data.empty:
            raise ValueError(
                "Input data is empty. Cannot make predictions on an empty DataFrame."
            )

        # Load pipeline
        if pm is None:
            pm = PipelineManagement()
        pipeline, processor = pm.load_pipeline()

        # Process test data using loaded processor
        X_test = test_data.drop(columns=[DEPENDENT_VAR_NAME])
        y_test = test_data[DEPENDENT_VAR_NAME]
        X_test_prcsd, y_test_prcsd = processor.transform(X_test, y_test)

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


if __name__ == "__main__":
    # Load data
    data = pd.read_csv("data/raw/validation.csv")
    pm = PipelineManagement()
    results = make_prediction(data, pm)
    print(results)
