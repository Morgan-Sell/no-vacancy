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
from db.postgres import AsyncPostgresDB
from schemas import bronze, gold, silver


# Initiate the database connections
bronze_db = AsyncPostgresDB(
    user=DB_USER,
    password=DB_PASSWORD,
    host=BRONZE_DB_HOST,
    port=DB_PORT,
    db_name=BRONZE_DB,
)

silver_db = AsyncPostgresDB(
    user=DB_USER,
    password=DB_PASSWORD,
    host=SILVER_DB_HOST,
    port=DB_PORT,
    db_name=SILVER_DB,
)

gold_db = AsyncPostgresDB(
    user=DB_USER,
    password=DB_PASSWORD,
    host=GOLD_DB_HOST,
    port=DB_PORT,
    db_name=GOLD_DB,
)

test_db = AsyncPostgresDB(
    user=TEST_DB_USER,
    password=TEST_DB_PASSWORD,
    host=TEST_DB_HOST,
    port=DB_PORT,
    db_name=TEST_DB,
)


# Create all tables if models are defined
async def init_all_databases():
    async with bronze_db.engine.begin() as conn:
        await conn.run_sync(bronze.Base.metadata.create_all)
  
    async with silver_db.engine.begin() as conn:
        await conn.run_sync(silver.Base.metadata.create_all)
    
    async with gold_db.engine.begin() as conn:
        await conn.run_sync(gold.Base.metadata.create_all)
    
    async with test_db.engine.begin() as conn:
        await conn.run_sync(bronze.Base.metadata.create_all)