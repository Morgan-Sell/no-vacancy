import logging
import warnings

import pandas as pd
from feature_engine.encoding import OneHotEncoder
from feature_engine.imputation import CategoricalImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from app.config import __model_version__
from app.services.config_services import (
    BOOKING_MAP,
    DATA_PATHS,
    MONTH_ABBREVIATION_MAP,
    SEARCH_SPACE,
    TARGET_VARIABLE,
    TRAIN_RATIO,
    VARIABLE_RENAME_MAP,
    VARS_TO_DROP,
    VARS_TO_IMPUTE,
    VARS_TO_OHE,
)
from app.services.pipeline import NoVacancyPipeline
from app.services.preprocessing import NoVacancyDataProcessing

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


def train_pipeline():
    # Load data
    data = pd.read_csv(DATA_PATHS["raw_data"])
    X = data.drop(columns=[TARGET_VARIABLE])
    y = data[TARGET_VARIABLE]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1 - TRAIN_RATIO, random_state=33
    )

    # Define pipeline components
    processor = NoVacancyDataProcessing(
        variable_rename=VARIABLE_RENAME_MAP,
        month_abbreviation=MONTH_ABBREVIATION_MAP,
        vars_to_drop=VARS_TO_DROP,
        booking_map=BOOKING_MAP,
    )
    imputer = CategoricalImputer(imputation_method="frequent", variables=VARS_TO_IMPUTE)
    encoder = OneHotEncoder(variables=VARS_TO_OHE)
    clsfr = RandomForestClassifier()

    # Train, finetune & test pipeline
    pipe = NoVacancyPipeline(processor, imputer, encoder, clsfr)
    pipe.pipeline(SEARCH_SPACE)
    pipe.fit(X_train, y_train)

    y_probs = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_probs)
    print("AUC: ", round(auc, 5))

    logger.info(f"{__model_version__} - AUC: {auc}")


if __name__ == "__main__":
    train_pipeline()
