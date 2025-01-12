# import simplejson
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
import pandas as pd
from pydantic import BaseModel

from app.config import __model_version__, get_logger
from app.services.data_management import DataManagement
from app.services.predictor import make_prediction

# Initalize logger
_logger = get_logger(logger_name=__name__)

# Define the router
router = APIRouter(prefix="/predict", tags=["predict"])


# Pydantic model for input validation
class PredictRequest(BaseModel):
    data: dict


@router.post("/", response_model=dict)
def predict(request_data: PredictRequest) -> dict:
    try: 
        # Convert the request data to a pandas DataFrame
        test_data = pd.DataFrame(request_data.data)

        # Ensure data management and predictions can run
        dm = DataManagement()
        results = make_prediction(test_data, dm)
    
        # Convert results to dictionary format for JSON response
        return {
            "predictions": results["prediction"].tolist(),
            "probability_not_canceled": results["probability_not_canceled"].tolist(),
            "probabilities_canceled": results["probabilities_canceled"].tolist(),
        }
    
    except HTTPException as http_exc:
        raise http_exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

