import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler
from os.path import abspath, dirname, join

from sqlalchemy import Column, Date, Float, Integer, String, create_engine


from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# -- Logging Config --
PACKAGE_ROOT = abspath(dirname(__file__))

FORMATTER = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s -" "%(funcName)s:%(lineno)d - %(message)s"
)
LOG_DIR = join(PACKAGE_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = join(LOG_DIR, "no_vacancy_api.log")

# -- Data Management --
PIPELINE_DIR = "app/models"
PIPELINE_SAVE_FILE = "no_vacancy_pipeline.pkl"

# Versioning
__api_version__ = "0.0.0"
__model_version__ = "0.0.3"


# Create a custom logger
def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler


def get_file_handler():
    file_handler = TimedRotatingFileHandler(LOG_FILE, when="midnight")
    file_handler.setFormatter(FORMATTER)
    file_handler.setLevel(logging.WARNING)
    return file_handler


def get_logger(*, logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.addHandler(get_console_handler())
    logger.addHandler(get_file_handler())
    logger.propagate = False
    return logger


# -- Postgres Config (Production) --
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("POSTGRES_PORT") # TODO: What to do with Postgres port (5432)?
DB_NAME = os.getenv("POSTGRES_DB") # TODO: Do I keep a "master" database?
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_CONNECT_TIMEOUT = 120  # in seconds

# Medallion DBs
BRONZE_DB_PORT = os.getenv("BRONZE_DB_PORT")
SILVER_DB_PORT = os.getenv("SILVER_DB_PORT")
GOLD_DB_PORT = os.getenv("GOLD_DB_PORT")

# CSV File Paths
DATA_DIR = "./data"
RAW_DATA_FILE_PATH = os.path.join(DATA_DIR, "bookings_raw.csv")


# Storage
CSV_HASH_TABLE = "csv_hashes"
RAW_DATA_TABLE = "raw_data"
