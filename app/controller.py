from fastapi import FastAPI
from app.config import get_logger
from app.version import __api_version__, __model_version__


# Initiliaze the FastAPI app
app = FastAPI()

_logger = get_logger(logger_name=__name__)

@app.get("/health")
async def health():
    """
    Health check endpoint
    """
    _logger.info("Health status OK")
    return {"status": "ok"}


@app.route("/version")
async def version():
    """
    API and model version endpoint.
    """
    return {
        "api_version": __api_version__,
        "model_version": __model_version__,
    }
