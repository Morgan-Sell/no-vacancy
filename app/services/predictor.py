import asyncio
import logging
import os
from datetime import datetime

import joblib
import mlflow
import numpy as np
import pandas as pd
from config import MLFLOW_TRACKING_URI, __model_version__
from db.db_init import gold_db, silver_db
from schemas.gold import Predictions
from schemas.silver import TestData
from services import DEPENDENT_VAR_NAME, MLFLOW_EXPERIMENT_NAME, MLFLOW_PROCESSOR_PATH
from services.pipeline_management import PipelineManagement
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.future import select

logger = logging.getLogger(__name__)


def handle_error(error_type, message, exception):
    logger.error(f"{message}: {exception}")
    raise error_type(f"{message}: {exception}")  # from exception


async def load_test_data(session_local, data_table):
    try:
        async with session_local as session:
            result = await session.execute(select(data_table))
            rows = result.scalars().all()
            if not rows:
                raise ValueError(f"No data found in the {data_table.__name__} table.")

            # Convert ORM objects to DataFrame
            df = pd.DataFrame([row.__dict__ for row in rows])
            df.drop(columns=["_sa_instance_state"], inplace=True)
            return df

    except SQLAlchemyError as db_err:
        handle_error(
            SQLAlchemyError,
            f"❌ Database error while loading the {data_table.__name__} table",
            db_err,
        )
    except Exception as e:
        handle_error(
            RuntimeError,
            f"❌ Unexpected error while loading the {data_table.__name__} table",
            e,
        )


def load_pipeline_and_processor_from_mlflow(stage: str = "Production"):

    # Use os.getenv to improve testability and flexibility for CLI overrides
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI))
    client = mlflow.MlflowClient()

    # Get the latest run in provided stage
    versions = client.get_latest_versions(MLFLOW_EXPERIMENT_NAME, stages=[stage])
    if not versions:
        raise RuntimeError(
            f"No model version found for stage '{stage}' in experiment '{MLFLOW_EXPERIMENT_NAME}'"
        )

    run_id = versions[0].run_id

    # Load pipeline
    model_uri = f"models:/{MLFLOW_EXPERIMENT_NAME}/{stage}"
    pipeline = mlflow.sklearn.load_model(model_uri=model_uri)

    # Load processor artifact
    local_processor_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=MLFLOW_PROCESSOR_PATH
    )
    processor = joblib.load(local_processor_path)

    return pipeline, processor


async def make_prediction(test_data: pd.DataFrame, pm: PipelineManagement = None):
    try:
        if not isinstance(test_data, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        if test_data.empty:
            raise ValueError(
                "Input data is empty. Cannot make predictions on an empty DataFrame."
            )

        # Load pipeline & processor artifacts
        pipeline, processor = load_pipeline_and_processor_from_mlflow(
            stage="Production"
        )

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
                "booking_id": test_data["booking_id"],
                "prediction": predictions,
                "probability_not_canceled": probabilities[:, 0],
                "probabilities_canceled": probabilities[:, 1],
            }
        )

        # Save predictions to the database
        async with gold_db.SessionLocal() as session:
            try:
                for _, row in results.iterrows():
                    pred_row = Predictions(
                        booking_id=row["booking_id"],
                        prediction=int(row["prediction"]),
                        probability_not_canceled=float(row["probability_not_canceled"]),
                        probability_canceled=float(row["probabilities_canceled"]),
                        model_version=__model_version__,
                        created_at=datetime.now(),
                    )
                    # Upsert the prediction into the database
                    await session.merge(pred_row)
                await session.commit()
            except SQLAlchemyError as db_err:
                await session.rollback()
                handle_error(
                    SQLAlchemyError,
                    "❌ Failed to save predictions to the database",
                    db_err,
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
    pm = PipelineManagement()

    async def run():
        data = await load_test_data(silver_db.SessionLocal(), TestData)
        await make_prediction(data, pm)

    asyncio.run(run())
