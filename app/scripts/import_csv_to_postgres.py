import csv
import hashlib
import os
import re
import socket
import time
from datetime import datetime

import psycopg2
from config import (
    BRONZE_DB,
    BRONZE_DB_HOST,
    CSV_HASH_TABLE,
    DB_CONNECT_TIMEOUT,
    DB_PASSWORD,
    DB_PORT,
    DB_USER,
    GOLD_DB_HOST,
    RAW_DATA_FILE_PATH,
    RAW_DATA_TABLE,
    SILVER_DB_HOST,
    TEST_DB_HOST,
)
from db.db_init import init_all_databases


def hash_csv(file_path):
    """Generate a SHA256 hash of the CSV file."""
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def get_csv_row_count(file_path):
    """Get the number of rows in the CSV file (excluding header)."""
    with open(file_path, "r") as f:
        return sum(1 for _ in f) - 1  # Exclude header row


def get_db_row_count(conn, table_name):
    """Get the number of rows in the database table."""
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {table_name};")
        return cur.fetchone()[0]


def create_log_table(conn):
    """Create the CSV hash log table if it doesn't exist."""
    with conn.cursor() as cur:
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {CSV_HASH_TABLE} (
                id SERIAL PRIMARY KEY,
                filename TEXT NOT NULL,
                file_hash TEXT NOT NULL,
                csv_row_count INTEGER NOT NULL,
                db_row_count INTEGER NOT NULL,
                imported_date TIMESTAMP DEFAULT NOW()
            );
        """
        )
    conn.commit()


def has_been_imported(conn, filename, file_hash):
    """Check if the file with this hash was already imported."""
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT 1 FROM {CSV_HASH_TABLE} 
            WHERE filename = %s
            AND file_hash = %s;
            """,
            (filename, file_hash),
        )
        return cur.fetchone() is not None


def log_import(conn, filename, file_hash, csv_rows, db_rows):
    """Always insert a new import log entry."""
    with conn.cursor() as cur:
        cur.execute(
            f"""
            INSERT INTO {CSV_HASH_TABLE} 
            (filename, file_hash, csv_row_count, db_row_count, imported_date)
            VALUES (%s, %s, %s, %s, %s);
        """,
            (filename, file_hash, csv_rows, db_rows, datetime.now()),
        )
    conn.commit()


def normalize_column_names(name):
    """Normalize column names to lowercase and replace spaces with underscores."""
    return re.sub(r"[^a-z0-9_]", "_", name.strip().lower())


def import_csv(conn, csv_file, table_name):
    """Import CSV file into the specified PostgreSQL table."""
    with conn.cursor() as cur, open(csv_file, "r") as f:
        reader = csv.reader(f)
        raw_header = next(reader)  # extracs header from CSV
        header = [normalize_column_names(col) for col in raw_header]

        placeholders = ", ".join(["%s"] * len(header))
        columns = ", ".join(header)

        for row in reader:
            # Ensures that values are only inserted into the columns that are
            # present in the CSV file
            cur.execute(
                f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders});", row
            )
    conn.commit()
    print(f"✅ Imported data from {csv_file} into {table_name}")


def wait_for_db(host, port, timeout=30):
    """Wait for the PostgreSQL database to be available."""
    start = time.time()

    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=3):
                print(f"✅ {host}:{port} is available.")
                return
        except OSError:
            print(f"Waiting for {host}:{port} to be available...")
            time.sleep(3)

    raise TimeoutError(f"❌ Timed out waiting for {host}:{port}")


def main():
    # Wait for all DBs to be available
    wait_for_db(BRONZE_DB_HOST, DB_PORT)
    wait_for_db(SILVER_DB_HOST, DB_PORT)
    wait_for_db(GOLD_DB_HOST, DB_PORT)
    wait_for_db(TEST_DB_HOST, DB_PORT)

    # Create tables if needed
    init_all_databases()

    # Connect the Bronze DB to perform SQL operations
    conn_bronze = psycopg2.connect(
        host=BRONZE_DB_HOST,
        port=DB_PORT,
        dbname=BRONZE_DB,
        user=DB_USER,
        password=DB_PASSWORD,
        connect_timeout=DB_CONNECT_TIMEOUT,
    )

    # Only create the log table if it doesn't already exist
    create_log_table(conn_bronze)

    file_hash = hash_csv(RAW_DATA_FILE_PATH)
    csv_rows = get_csv_row_count(RAW_DATA_FILE_PATH)

    if has_been_imported(conn_bronze, RAW_DATA_FILE_PATH, file_hash):
        db_rows = get_db_row_count(conn_bronze, RAW_DATA_TABLE)
        log_import(conn_bronze, RAW_DATA_FILE_PATH, file_hash, csv_rows, db_rows)
    else:
        db_row_before_import = get_db_row_count(conn_bronze, RAW_DATA_TABLE)
        import_csv(conn_bronze, RAW_DATA_FILE_PATH, RAW_DATA_TABLE)
        db_row_after_import = get_db_row_count(conn_bronze, RAW_DATA_TABLE)
        log_import(
            conn_bronze,
            RAW_DATA_FILE_PATH,
            file_hash,
            csv_rows,
            db_row_after_import - db_row_before_import,
        )

        print(
            f"✅ Successfully imported {csv_rows} rows from {RAW_DATA_FILE_PATH} into {RAW_DATA_TABLE}"
        )

    conn_bronze.close()


if __name__ == "__main__":
    main()
