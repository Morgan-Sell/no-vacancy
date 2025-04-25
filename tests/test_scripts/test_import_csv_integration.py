import hashlib
import os

import psycopg2
import pytest
from scripts.import_csv_to_postgres import (
    get_csv_row_count,
    get_db_row_count,
    has_been_imported,
    hash_csv,
    import_csv,
    log_import,
)

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
from tests.conftest import TEST_TABLE


def test_hash_csv_and_row_count(temp_booking_data_csv):
    """Test CSV having and row count calculation"""
    with open(temp_booking_data_csv, "rb") as f:
        expected_hash = hashlib.sha256(f.read()).hexdigest()

    computed_hash = hash_csv(temp_booking_data_csv)
    assert computed_hash == expected_hash

    row_count = get_csv_row_count(temp_booking_data_csv)
    assert row_count == 20


def test_import_csv_and_log(temp_booking_data_csv, test_db_conn):
    """Test CSV import and import log functionality."""
    # Filter CSV to match test table schema
    with open(temp_booking_data_csv, "r") as f:
        header = f.readline().strip().split(",")
        filtered_csv_path = temp_booking_data_csv.replace(".csv", "_filtered.csv")
        with open(filtered_csv_path, "w") as filtered_csv:
            filtered_csv.write("number_of_adults,number_of_weekend_nights\n")
            for line in f:
                cols = line.strip().split(",")
                indices = [
                    header.index("number of adults"),
                    header.index("number of weekend nights"),
                ]
                filtered_csv.write(f"{cols[indices[0]]},{cols[indices[1]]}\n")

    file_hash = hash_csv(filtered_csv_path)
    csv_rows = get_csv_row_count(filtered_csv_path)

    # Ensure this version hasn't been imported
    assert not has_been_imported(
        test_db_conn, filtered_csv_path, file_hash
    ), "CSV should be not marked as imported"

    # Import CSV into test table
    import_csv(test_db_conn, filtered_csv_path, TEST_TABLE)
    db_rows = get_db_row_count(test_db_conn, TEST_TABLE)
    assert db_rows == csv_rows, "DB row count should match CSV row count"

    # Log the import
    log_import(test_db_conn, filtered_csv_path, file_hash, csv_rows, db_rows)
    assert has_been_imported(
        test_db_conn, filtered_csv_path, file_hash
    ), "CSV should be marked as imported"

    # Clean up temp file
    os.remove(filtered_csv_path)
