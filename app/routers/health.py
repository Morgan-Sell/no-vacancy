from config import get_logger
from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter(
    prefix="/health",
    tags=["health"],
)

_logger = get_logger(logger_name=__name__)


@router.get("/", response_class=JSONResponse)
def health():
    """
    Health check endpoint
    """
    _logger.info("Health status OK")
    return {"status": "ok"}
