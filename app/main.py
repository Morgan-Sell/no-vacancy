from fastapi import FastAPI
from app.api.routes import healthcheck, predictions
from app.core.logger import setup_logger

# Initialize the logger
setup_logger()

# Create the FastAPI app
app = FastAPI(
    title="Hotel Reservation Prediction API",
    description="An API to predict the likelihood of someone cancelling their hotel reservation",
    version="0.0.0"
)

# 