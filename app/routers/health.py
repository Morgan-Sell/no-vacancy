from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from app.config import get_logger

router = APIRouter(
    prefix="/health",
    tags=["health"],
)

_logger = get_logger(logger_name=__name__)


@router.post("/", response_class=dict)
async def health():
    """
    Health check endpoint
    """
    _logger.info("Health status OK")
    return {"status": "ok"}
