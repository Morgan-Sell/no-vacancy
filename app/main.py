from fastapi import APIRouter, FastAPI

from app.config import get_logger
from app.endpoints import health, predict, version

_logger = get_logger(logger_name=__name__)

app = FastAPI()

router = APIRouter()

@app.get("/")
async def root():
    return {"message": "Welcome to the No Vacancy API!"}

# Add endpoints to router
router.add_api_route("/health", health.health, methods=["GET"], tags=["health"])
router.add_api_route("/predict", predict.predict, methods=["POST"], tags=["predict"])
router.add_api_route("/version", version.version, methods=["GET"], tags=["version"])

# Add router to app
app.include_router(router)