import logging
import warnings

import pandas as pd
from feature_engine.encoding import OneHotEncoder
from feature_engine.imputation import CategoricalImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from config import __model_version__
from services import (
    BOOKING_MAP,
    DATA_PATHS,
    IMPUTATION_METHOD,
    MONTH_ABBREVIATION_MAP,
    SEARCH_SPACE,
    TARGET_VARIABLE,
    TRAIN_RATIO,
    VARIABLE_RENAME_MAP,
    VARS_TO_DROP,
    VARS_TO_IMPUTE,
    VARS_TO_OHE,
)
from services.pipeline_management import PipelineManagement
from services.pipeline import NoVacancyPipeline
from services.preprocessing import NoVacancyDataProcessing

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


def train_pipeline():
    # Load data
    data = pd.read_csv(
        DATA_PATHS["raw_data"]
    )  # TODO: Need to update when data storage is added

    # Split data
    X = data.drop(columns=[TARGET_VARIABLE])
    y = data[TARGET_VARIABLE]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1 - TRAIN_RATIO, random_state=33
    )

    # Preprocess data separately b/c Pipeline() does not handle target variable transformation
    # Also datetime object needs to be dropped before pipe.fit()
    processor = NoVacancyDataProcessing(
        variable_rename=VARIABLE_RENAME_MAP,
        month_abbreviation=MONTH_ABBREVIATION_MAP,
        vars_to_drop=VARS_TO_DROP,
        booking_map=BOOKING_MAP,
    )
    X_train_prcsd, y_train_prcsd = processor.fit_transform(X_train, y_train)
    X_test_prcsd, y_test_prcsd = processor.transform(X_test, y_test)

    # Define pipeline components
    imputer = CategoricalImputer(imputation_method=IMPUTATION_METHOD, variables=VARS_TO_IMPUTE)
    encoder = OneHotEncoder(variables=VARS_TO_OHE)
    clsfr = RandomForestClassifier()


    # print("[DEBUG] X_train_prcsd columns: ", X_train_prcsd.columns)   
    # print("\n[DEBUG] X_test_prcsd columns: ", X_test_prcsd.columns)

    # Train, finetune & test pipeline
    pipe = NoVacancyPipeline(imputer, encoder, clsfr)
    pipe.fit(X_train_prcsd, y_train_prcsd, search_space=SEARCH_SPACE)

    # Save the pipeline and processor
    pm = PipelineManagement()
    pm.save_pipeline(pipe, processor)

    # Perform predictions and evaluate performance
    y_probs = pipe.predict_proba(X_test_prcsd)[:, 1]
    auc = roc_auc_score(y_test_prcsd, y_probs)
    print("AUC: ", round(auc, 5))
    logger.info(f"{__model_version__} - AUC: {auc}")


if __name__ == "__main__":
    train_pipeline()
