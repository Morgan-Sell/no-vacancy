from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from app.config import __api_version__, __model_version__


async def version():
    """
    API and model version endpoint.
    """
    return {
        "api_version": __api_version__,
        "model_version": __model_version__,
    }
