import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config import (
    BRONZE_DB,
    BRONZE_DB_PORT,
    DB_CONNECT_TIMEOUT,
    BRONZE_DB_HOST,
    SILVER_DB_HOST,
    GOLD_DB_HOST,
    DB_PASSWORD,
    DB_USER,
    ENV,
    GOLD_DB,
    GOLD_DB_PORT,
    SILVER_DB,
    SILVER_DB_PORT,
    TEST_BRONZE_DB,
    TEST_BRONZE_DB_PORT,
    TEST_BRONZE_DB_HOST,
    TEST_SILVER_DB_HOST,
    TEST_GOLD_DB_HOST,
    TEST_DB_PASSWORD,
    TEST_DB_USER,
    TEST_GOLD_DB,
    TEST_GOLD_DB_PORT,
    TEST_SILVER_DB,
    TEST_SILVER_DB_PORT,
)

IS_PROD = ENV == "PROD"


def make_postgres_url(user, password, host, port, db_name):
    return f"postgresql://{user}:{password}@{host}:{port}/{db_name}"


# Create Postgres URLs depending on environment
BRONZE_DB_URL = make_postgres_url(
    DB_USER if IS_PROD else TEST_DB_USER,
    DB_PASSWORD if IS_PROD else TEST_DB_PASSWORD,
    BRONZE_DB_HOST if IS_PROD else TEST_BRONZE_DB_HOST,
    BRONZE_DB_PORT if IS_PROD else TEST_BRONZE_DB_PORT,
    BRONZE_DB if IS_PROD else TEST_BRONZE_DB,
)

SILVER_DB_URL = make_postgres_url(
    DB_USER if IS_PROD else TEST_DB_USER,
    DB_PASSWORD if IS_PROD else TEST_DB_PASSWORD,
    SILVER_DB_HOST if IS_PROD else TEST_SILVER_DB_HOST,
    SILVER_DB_PORT if IS_PROD else TEST_SILVER_DB_PORT,
    SILVER_DB if IS_PROD else TEST_SILVER_DB,
)

GOLD_DB_URL = make_postgres_url(
    DB_USER if IS_PROD else TEST_DB_USER,
    DB_PASSWORD if IS_PROD else TEST_DB_PASSWORD,
    GOLD_DB_HOST if IS_PROD else TEST_GOLD_DB_HOST,
    GOLD_DB_PORT if IS_PROD else TEST_GOLD_DB_PORT,
    GOLD_DB if IS_PROD else TEST_GOLD_DB,
)


# -- Bronze DB --
bronze_engine = create_engine(
    BRONZE_DB_URL, connect_args={"connect_timeout": DB_CONNECT_TIMEOUT}
)
# autocommit and authoflush set to false to ensure atomicity
BronzeSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=bronze_engine)

# -- Silver DB --
silver_engine = create_engine(
    SILVER_DB_URL, connect_args={"connect_timeout": DB_CONNECT_TIMEOUT}
)
# autocommit and authoflush set to false to ensure atomicity
SilverSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=silver_engine)

# -- Gold DB --
gold_engine = create_engine(
    GOLD_DB_URL, connect_args={"connect_timeout": DB_CONNECT_TIMEOUT}
)
# autocommit and authoflush set to false to ensure atomicity
GoldSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=gold_engine)


## -- Create all tables if models are defined --
def init_all_databases():
    from app.schemas import bronze, gold, silver

    bronze.Base.metadata.create_all(bind=bronze_engine)
    silver.Base.metadata.create_all(bind=silver_engine)
    gold.Base.metadata.create_all(bind=gold_engine)
