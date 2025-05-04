import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler
from os.path import abspath, dirname, join

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


# -- General Confg --
# Determin environment (test, prod)
ENV = os.getenv("ENV", "PROD").upper()

# -- Postgres Config (Production) --
# General DB config

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_CONNECT_TIMEOUT = 120  # in seconds
DB_PORT = os.getenv("DB_PORT")

# Bronze DB config
BRONZE_DB_HOST = os.getenv("BRONZE_DB_HOST")
BRONZE_DB = os.getenv("BRONZE_DB")
BRONZE_DB_PORT = os.getenv("BRONZE_DB_PORT")

# Silver DB config
SILVER_DB_HOST = os.getenv("SILVER_DB_HOST")
SILVER_DB = os.getenv("SILVER_DB")
SILVER_DB_PORT = os.getenv("SILVER_DB_PORT")


# Gold DB config
GOLD_DB_HOST = os.getenv("GOLD_DB_HOST")
GOLD_DB = os.getenv("GOLD_DB")
GOLD_DB_PORT = os.getenv("GOLD_DB_PORT")

# Test DB config
TEST_DB_USER = os.getenv("TEST_DB_USER")
TEST_DB_PASSWORD = os.getenv("TEST_DB_PASSWORD")
TEST_DB_HOST = os.getenv("TEST_DB_HOST")
TEST_DB = os.getenv("TEST_DB")
TEST_DB_PORT = os.getenv("TEST_DB_PORT")


# CSV File Paths
DATA_DIR = "./data"
RAW_DATA_FILE_PATH = os.path.join(DATA_DIR, "bookings_raw.csv")


# Storage
CSV_HASH_TABLE = "csv_hashes"
RAW_DATA_TABLE = "raw_data"
