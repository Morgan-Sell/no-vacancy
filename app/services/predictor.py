import logging

import pandas as pd

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


def make_prediction(test_data: pd.DataFrame):

    try:
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

        # Format predictions into a dataframe
        results = pd.DataFrame(
            {
                "prediction": predictions,
                "probability_not_canceled": probabilities[:, 0],
                "probabilities_canceled": probabilities[:, 1],
            }
        )

        return results

    except FileNotFoundError as e:
        logger.error(f"❌ No pipeline found: {e}")
        raise RuntimeError(f"❌ No pipeline found: {e}")
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        raise RuntimeError(f"❌ Prediction failed: {e}")
