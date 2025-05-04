from config import (
    BRONZE_DB,
    BRONZE_DB_HOST,
    DB_PASSWORD,
    DB_PORT,
    DB_USER,
    GOLD_DB,
    GOLD_DB_HOST,
    SILVER_DB,
    SILVER_DB_HOST,
    TEST_DB,
    TEST_DB_HOST,
    TEST_DB_PASSWORD,
    TEST_DB_USER,
)
from db.postgres import PostgresDB

# Initiate the database connections
bronze_db = PostgresDB(
    user=DB_USER,
    password=DB_PASSWORD,
    host=BRONZE_DB_HOST,
    port=DB_PORT,
    db_name=BRONZE_DB,
)

silver_db = PostgresDB(
    user=DB_USER,
    password=DB_PASSWORD,
    host=SILVER_DB_HOST,
    port=DB_PORT,
    db_name=SILVER_DB,
)

gold_db = PostgresDB(
    user=DB_USER,
    password=DB_PASSWORD,
    host=GOLD_DB_HOST,
    port=DB_PORT,
    db_name=GOLD_DB,
)

test_db = PostgresDB(
    user=TEST_DB_USER,
    password=TEST_DB_PASSWORD,
    host=TEST_DB_HOST,
    port=DB_PORT,
    db_name=TEST_DB,
)


# Create all tables if models are defined
def init_all_databases():
    from schemas import bronze, gold, silver

    # Production databases
    bronze.Base.metadata.create_all(bind=bronze_db.engine)
    silver.Base.metadata.create_all(bind=silver_db.engine)
    gold.Base.metadata.create_all(bind=gold_db.engine)

    # Use for testing
    bronze.Base.metadata.create_all(bind=test_db.engine)
