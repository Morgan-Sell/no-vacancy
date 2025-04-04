from app.config import get_logger
from fastapi import FastAPI
from app.routers import health, predict, version

_logger = get_logger(logger_name=__name__)

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Welcome to the No Vacancy API!"}


# Register routers with API instance
app.include_router(health.router)
app.include_router(predict.router)
app.include_router(version.router)
