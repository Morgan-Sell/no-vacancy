# import simplejson
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
import pandas as pd
from pydantic import BaseModel
from requests import request

from app.config import __model_version__, get_logger
from app.services.pipeline_management import PipelineManagement
from app.services.predictor import make_prediction

# Define the router
router = APIRouter(prefix="/predict", tags=["predict"])

# Initalize logger
_logger = get_logger(logger_name=__name__)


# Pydantic model for input validation
class PredictionRequest(BaseModel):
    data: list[dict]

@router.post("/", response_model=dict)
def predict(request_data: PredictionRequest) -> dict:
    try:
        # Log the received input
        _logger.debug(f"Inputs: {request_data}")

        # Convert the input data to a pandas DataFrame
        test_data = pd.DataFrame(request_data.data)

        # Ensure predictions can be made
        dm = PipelineManagement()
        results = make_prediction(test_data, dm)

        # Extract predictions and version
        predictions = results["prediction"].to_list()
        version = results["version"]

        # FastAPI automatically converts Python data structures into JSON responses
        return {
            "predictions": predictions,
            "version": version,
        } 
    
    except Exception as exc:
        _logger.error(f"Unexpected error during prediction: {exc}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")