import logging
import warnings

import pandas as pd
from config import __model_version__
from feature_engine.encoding import OneHotEncoder
from feature_engine.imputation import CategoricalImputer
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
from services.pipeline import NoVacancyPipeline
from services.pipeline_management import PipelineManagement
from services.preprocessing import NoVacancyDataProcessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from db.postgres import BronzeSessionLocal, SilverSessionLocal
from schemas.bronze import RawData
from schemas.silver import TrainData, ValidationTestData

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


def load_raw_data(session: BronzeSessionLocal):
    """
    Load raw data from the Bronze database.
    """
    records = session.query(RawData).all()
    # Convert list of SQLAlchemy model instances to a dataframe
    df = pd.DataFrame([record.__dict__ for record in records])
    # Remove SQLAchemy metadata
    df.drop(columns=["_sa_instance_state"], inplace=True)
    return df


def preprocess_data(X, y, processor):
    """
    Preprocess the data using the NoVacancyDataProcessing transformer.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1 - TRAIN_RATIO, random_state=33
    )
    X_train_prcsd, y_train_prcsd = processor.fit_transform(X_train, y_train)
    X_test_prcsd, y_test_prcsd = processor.transform(X_test, y_test)
    return X_train_prcsd, X_test_prcsd, y_train_prcsd, y_test_prcsd


def save_to_silver_db(X_train, y_train, X_test, y_test, session: SilverSessionLocal):
    """
    Save the preprocessed data to the Silver database.
    """
    X_train["is_cancellation"] = y_train
    X_test["is_cancellation"] = y_test

    # Create a list of TrainData instances
    train_objects = [
        TrainData(**row._asdict()) for row in X_train.itertuples(index=False)
    ]
    # Create a list of ValidationTestData instances
    test_objects = [
        ValidationTestData(**row._asdict()) for row in X_test.itertuples(index=False)
    ]

    # Insert ORM objects to database w/o primary key updates and relationship handling
    session.bulk_save_objects(train_objects)
    session.bulk_save_objects(test_objects)
    session.commit()


def build_pipeline():
    imputer = CategoricalImputer(
        imputation_method=IMPUTATION_METHOD, variables=VARS_TO_IMPUTE
    )
    encoder = OneHotEncoder(variables=VARS_TO_OHE)
    clsfr = RandomForestClassifier()
    return NoVacancyPipeline(imputer, encoder, clsfr)


def evaluate_model(pipe, X_test, y_test):
    y_probs = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_probs)
    logger.info(f"{__model_version__} - AUC: {auc}")
    print("AUC: ", round(auc, 5))


def train_pipeline():
    # Preprocess data separately b/c Pipeline() does not handle target variable transformation
    # Also datetime object needs to be dropped before pipe.fit()
    processor = NoVacancyDataProcessing(
        variable_rename=VARIABLE_RENAME_MAP,
        month_abbreviation=MONTH_ABBREVIATION_MAP,
        vars_to_drop=VARS_TO_DROP,
        booking_map=BOOKING_MAP,
    )

    # Load and process raw data
    with BronzeSessionLocal() as bronze_session:
        df = load_raw_data(bronze_session)

    X = df.drop(columns=[TARGET_VARIABLE])
    y = df[TARGET_VARIABLE]
    X_train, X_test, y_train, y_test = preprocess_data(X, y, processor)

    # Save preprocessed data to Silver database
    with SilverSessionLocal() as silver_session:
        save_to_silver_db(X_train, y_train, X_test, y_test, silver_session)

    pipe = build_pipeline()
    pipe.fit(X_train, y_train, search_space=SEARCH_SPACE)

    # Save the pipeline and processor
    pm = PipelineManagement()
    pm.save_pipeline(pipe, processor)
    evaluate_model(pipe, X_test, y_test)


if __name__ == "__main__":
    train_pipeline()
