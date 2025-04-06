import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler
from os.path import abspath, dirname, join

from sqlalchemy import Column, Date, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

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
DB_CONNECT_TIMEOUT = 120  # in seconds

# CSV File Paths
DATA_DIR = "./data/raw"
TRAIN_CSV_FILE_PATH = os.path.join(DATA_DIR, "train.csv")
VALIDATION_CSV_FILE_PATH = os.path.join(DATA_DIR, "validation.csv")
TEST_CSV_FILE_PATH = os.path.join(DATA_DIR, "test.csv")

# Postgres Data Tables
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(
    DATABASE_URL,
    connect_args={"connect_timeout": DB_CONNECT_TIMEOUT}
)

# autocommit and authoflush set to false to ensure atomicity
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for ORM models
Base = declarative_base()


class Bronze(Base):
    """
    Bronze table for raw data storage.
    """
    __tablename__ = "bronze"

    booking_id = Column(String, primary_key=True)
    number_of_adults = Column(Integer, nullable=True)
    number_of_children = Column(Integer, nullable=True)
    number_of_weekend_nights = Column(Integer, nullable=True)
    number_of_weekdays_nights = Column(Integer, nullable=True)
    type_of_meal = Column(String, nullable=True)
    car_parking_space = Column(Integer, nullable=True)
    room_type = Column(String, nullable=True)
    lead_time = Column(Integer, nullable=True)  # in days
    market_segment_type = Column(String, nullable=True)
    is_repeat_guest = Column(Integer, nullable=True)  # 1 if repeat guest, 0 otherwise
    num_previous_cancellations = Column(Integer, nullable=True)  
    num_previous_bookings_not_canceled = Column(Integer, nullable=True)
    average_price = Column(Float, nullable=True)
    special_requests = Column(Integer, nullable=True)
    date_of_reservation = Column(Date, nullable=True)
    booking_stagus = Column(String, nullable=False)


class Silver(Base):
    """
    Silver table for processed data storage.
    """
    __tablename__ = "silver"

    number_of_weekend_nights = Column(Integer, nullable=False)
    number_of_weekdays_nights = Column(Integer, nullable=False)
    lead_time = Column(Integer, nullable=False)  # in days
    type_of_meal = Column(String, nullable=False)
    car_parking_space = Column(Integer, nullable=False)
    room_type = Column(String, nullable=False)
    average_price = Column(Float, nullable=False)
    is_type_of_meal_meal_plan_1 = Column(Integer, nullable=False)
    is_type_of_meal_meal_plan_2 = Column(Integer, nullable=False)
    is_type_of_meal_meal_plan_3 = Column(Integer, nullable=False)
    is_room_type_room_type_1 = Column(Integer, nullable=False)
    is_room_type_room_type_2 = Column(Integer, nullable=False)
    is_room_type_room_type_3 = Column(Integer, nullable=False)
    is_room_type_room_type_4 = Column(Integer, nullable=False)
    is_room_type_room_type_5 = Column(Integer, nullable=False)
    is_room_type_room_type_6 = Column(Integer, nullable=False)
    is_room_type_room_type_7 = Column(Integer, nullable=False)