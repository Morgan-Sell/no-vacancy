import logging

import numpy as np
import pandas as pd

from app.services import DATA_PATHS
from app.services.pipeline_management import PipelineManagement

logger = logging.getLogger(__name__)


def handle_error(error_type, message, exception):
    logger.error(f"{message}: {exception}")
    raise error_type(f"{message}: {exception}")  # from exception


def make_prediction(X_test: pd.DataFrame, pm: PipelineManagement):

    try:
        if not isinstance(X_test, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        if X_test.empty:
            raise ValueError(
                "Input data is empty. Cannot make predictions on an empty DataFrame."
            )

        # Load pipeline
        if pm is None:
            pm = PipelineManagement()
        pipeline, processor = pm.load_pipeline()

        # Process test data using loaded processor
        X_test_prcsd, _ = processor.transform(X_test)  # , y_test)

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
    pm = PipelineManagement(pipeline_path=DATA_PATHS["model_save_path"])
    results = make_prediction(data, pm)
    print(results)
