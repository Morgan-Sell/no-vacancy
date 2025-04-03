# import simplejson
import pandas as pd
from app.services import DATA_PATHS
from config import __model_version__, get_logger
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.pipeline_management import PipelineManagement
from services.predictor import make_prediction

# Define the router
router = APIRouter(prefix="/predict", tags=["predict"])

# Initalize logger
_logger = get_logger(logger_name=__name__)


# Pydantic model for input validation
class PredictionRequest(BaseModel):
    data: list[dict]


# Pydantic model for output validation
class PredictionResponse(BaseModel):
    predictions: list[float]
    version: str


@router.post("/", response_model=PredictionResponse)
def predict(request_data: PredictionRequest):
    try:
        # Log the received input
        _logger.debug(f"Inputs: {request_data}")

        # Convert the input data to a pandas DataFrame
        test_data = pd.DataFrame(request_data.data)

        # Ensure predictions can be made
        pm = PipelineManagement(DATA_PATHS["model_save_path"])
        results = make_prediction(test_data, pm)

        # Extract predictions and version
        predictions = results["prediction"].to_list()
        # version = results["version"]

        # FastAPI automatically converts Python data structures into JSON responses
        return {
            "predictions": predictions,
            "version": __model_version__,
        }

    except Exception as exc:
        _logger.error(f"Unexpected error during prediction: {exc}")
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {exc}"
        ) from exc
