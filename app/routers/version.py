from fastapi import APIRouter
from fastapi.responses import JSONResponse

from config import __api_version__, __model_version__

# Define the router
router = APIRouter(prefix="/version", tags=["version"])


@router.get("/", response_class=JSONResponse)
def version():
    """
    API and model version endpoint.
    """
    return {
        "api_version": __api_version__,
        "model_version": __model_version__,
    }
