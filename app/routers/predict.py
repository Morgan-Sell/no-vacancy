# import simplejson
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.config import get_logger, __model_version__

# Initalize logger
_logger = get_logger(logger_name=__name__)

# Define the router
router = APIRouter(prefix="/predict", tags=["predict"])


# Pydantic model for input validation
class PredictRequest(BaseModel):
    data: dict


# @router.post("/", response_model=dict)
# def predict(request_data: PredictRequest) -> dict:
#     try:
#         # Extract JSON payload
#         json_data = request_data.data
#         _logger.debut(f"Inputs: {json_data}")

#         # Normalize and prepare data
#         normalized_str = simplejson.dumps(json_data, ignore_nan=True)
#         data_js_dict = simplejson.loads(normalized_str)

#         # Make predictions
#         results = make_prediction(
#             test_data
#         )  # TODO: Create make_prediction in  PredictionService
#         predictions = results["predictions"].values.tolist()

#         # Return response
#         return {
#             "predictions": predictions,
#             "version": model_version,
#         }

#     except Exception as e:
#         _logger.error(f"Error in prediction: {e}")
#         raise HTTPException(status_code=500, detail="Prediction failed")
