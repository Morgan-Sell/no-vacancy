from fastapi import FastAPI

from app.config import get_logger
from app.routers import health, predict, version

_logger = get_logger(logger_name=__name__)

app = FastAPI()


@app.get("/")
async def root():
    pass


app.include_router(health.router)
app.include_router(version.router)
app.include_router(predict.router)
