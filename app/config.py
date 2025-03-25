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


# -- Postgres Config (Production) --
DB_HOST = os.getenv("POSTGRES_HOST")
DB_PORT = os.getenv("POSTGRES_PORT")
DB_NAME = os.getenv("POSTGRES_DB")
DB_USER = os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")

# CSV File Paths
DATA_DIR = "./data/raw"
TRAIN_CSV_FILE_PATH = os.path.join(DATA_DIR, "train.csv")
VALIDATION_CSV_FILE_PATH = os.path.join(DATA_DIR, "validation.csv")
TEST_CSV_FILE_PATH = os.path.join(DATA_DIR, "test.csv")

# Table Mappings: CSV -> Table Name
CSV_TABLE_MAP = {
    TRAIN_CSV_FILE_PATH: "train_table",
    VALIDATION_CSV_FILE_PATH: "validation_table",
    TEST_CSV_FILE_PATH: "test_table",
}
CSV_HASH_TABLE = "data_import_log"