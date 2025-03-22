import os

import psycopg2
import pytest

from app.config import (
    CSV_HASH_TABLE,
    CSV_TABLE_MAP,
    DB_HOST,
    DB_NAME,
    DB_PASSWORD,
    DB_PORT,
    DB_USER,
    TEST_CSV_FILE_PATH,
    TRAIN_CSV_FILE_PATH,
    VALIDATION_CSV_FILE_PATH,
)
from scripts.import_csv_to_postgres import (
    get_csv_row_count,
    get_db_row_count,
    has_been_imported,
    hash_csv,
    import_csv,
    log_import,
)

# Constants
TEST_TABLE = "test_import_table"


@pytest.fixture(scope="module")
def test_db_conn():
    """Connect to Postgres test DB."""
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )
    yield conn
    conn.close()
