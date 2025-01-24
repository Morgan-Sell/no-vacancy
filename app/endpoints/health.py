from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from app.config import get_logger

_logger = get_logger(logger_name=__name__)


async def health():
    """
    Health check endpoint
    """
    _logger.info("Health status OK")
    return {"status": "ok"}
