import asyncio
import os
import tempfile
import warnings
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from config import MLFLOW_TRACKING_URI, __model_version__, get_logger
from db.db_init import bronze_db, silver_db
from feature_engine.encoding import OneHotEncoder
from feature_engine.imputation import CategoricalImputer
from schemas.bronze import RawData
from schemas.silver import TestData, TrainValidationData
from services import (
    BOOKING_MAP,
    IMPUTATION_METHOD,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_PROCESSOR_JOBLIB,
    MONTH_ABBREVIATION_MAP,
    PRIMARY_KEY,
    RAW_TARGET_VARIABLE,
    SEARCH_SPACE,
    TRAIN_RATIO,
    VARIABLE_RENAME_MAP,
    VARS_TO_DROP,
    VARS_TO_IMPUTE,
    VARS_TO_OHE,
)
from services.pipeline import NoVacancyPipeline
from services.preprocessing import NoVacancyDataProcessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

logger = get_logger(logger_name=__name__)
warnings.filterwarnings("ignore")


async def load_raw_data(session: AsyncSession, table: RawData):
    """
    Load raw data from the Bronze database.
    """
    result = await session.execute(select(table))
    records = result.scalars().all()
    if not records:
        raise ValueError("No data found in the Bronze database.")
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


async def save_to_silver_db(X_train, y_train, X_test, y_test, session: AsyncSession):
    """
    Save the preprocessed data to the Silver database.
    """
    X_train["is_cancellation"] = y_train
    X_test["is_cancellation"] = y_test

    # Create a list of TrainData instances
    train_objects = [
        TrainValidationData(**row._asdict()) for row in X_train.itertuples(index=False)
    ]
    # Create a list of ValidationTestData instances
    test_objects = [TestData(**row._asdict()) for row in X_test.itertuples(index=False)]

    # Insert ORM objects to database w/o primary key updates and relationship handling
    session.add_all(train_objects)
    session.add_all(test_objects)
    await session.commit()


def build_pipeline():
    imputer = CategoricalImputer(
        imputation_method=IMPUTATION_METHOD,
        variables=VARS_TO_IMPUTE,
    )
    encoder = OneHotEncoder(variables=VARS_TO_OHE)
    clsfr = RandomForestClassifier()
    return NoVacancyPipeline(imputer, encoder, clsfr, [PRIMARY_KEY])


def evaluate_model(pipe, X_test, y_test):
    y_probs = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_probs)
    logger.info(f"{__model_version__} - AUC: {auc}")
    print("AUC: ", round(auc, 5))
    return auc


def setup_mlflow():
    # Use os.getenv to improve testability and flexibility for CLI overrides
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI))
    # If an experiment does not exist, MLflow will create it.
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


async def train_pipeline():
    # Preprocess data separately b/c Pipeline() does not handle target variable transformation
    # Also datetime object needs to be dropped before pipe.fit()
    processor = NoVacancyDataProcessing(
        variable_rename=VARIABLE_RENAME_MAP,
        month_abbreviation=MONTH_ABBREVIATION_MAP,
        vars_to_drop=VARS_TO_DROP,
        booking_map=BOOKING_MAP,
    )

    # Load and process raw data
    async with bronze_db.create_session()() as bronze_session:
        df = await load_raw_data(bronze_session, RawData)

    logger.info("✅ Loaded raw data")

    X = df.drop(columns=[RAW_TARGET_VARIABLE])
    y = df[RAW_TARGET_VARIABLE]
    X_train, X_test, y_train, y_test = preprocess_data(X, y, processor)

    # Save preprocessed data to Silver database
    async with silver_db.create_session()() as silver_session:
        await save_to_silver_db(X_train, y_train, X_test, y_test, silver_session)

    logger.info("✅ Saved preprocessed data to Silver database")

    # Build and train the pipeline
    pipe = build_pipeline()
    pipe.fit(X_train, y_train, search_space=SEARCH_SPACE)

    # Save the pipeline and processor
    X_test.drop(columns=[PRIMARY_KEY], inplace=True, errors="ignore")
    test_score = evaluate_model(pipe, X_test, y_test)

    # MLflow logging
    logged_params = pipe.get_logged_params()
    logged_params["model_version"] = __model_version__

    # Create parquet to save an input example
    X_test.iloc[:1].to_parquet("input_example.parquet", index=False)

    setup_mlflow()
    with mlflow.start_run():
        mlflow.log_params(logged_params)

        # Log model pipeline
        mlflow.sklearn.log_model(
            pipe.get_full_pipeline(),
            "model",
            signature=mlflow.models.infer_signature(X_test, pipe.predict(X_test)),
            registered_model_name=MLFLOW_EXPERIMENT_NAME,
        )

        # Log metrics
        mlflow.log_metric("val_auc", logged_params["best_model_val_score"])
        mlflow.log_metric("test_auc", test_score)

        # Create a temporary directory to save the processor
        # and log it as an artifact. Prevents overcrowding in project root.
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            processor_path = tmp_path / MLFLOW_PROCESSOR_JOBLIB
            joblib.dump(processor, processor_path)
            mlflow.log_artifact(str(processor_path), artifact_path="processor")

            # Saving input_example.parquet separately to mitigate risk of model logging slowdown
            input_example_path = tmp_path / "input_example.parquet"
            X_test.iloc[:1].to_parquet(input_example_path, index=False)
            mlflow.log_artifact(str(input_example_path), artifact_path="input_example")

    logger.info("✅ Pipeline and processor saved to MLflow")


if __name__ == "__main__":
    asyncio.run(train_pipeline())
