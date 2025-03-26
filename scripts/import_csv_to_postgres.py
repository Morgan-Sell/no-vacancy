import csv
import hashlib
import os
from datetime import datetime

import psycopg2

from app.config import (
    CSV_HASH_TABLE,
    CSV_TABLE_MAP,
    DB_CONNECT_TIMEOUT,
    DB_HOST,
    DB_NAME,
    DB_PASSWORD,
    DB_PORT,
    DB_USER,
    TEST_CSV_FILE_PATH,
    TRAIN_CSV_FILE_PATH,
    VALIDATION_CSV_FILE_PATH,
)


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


def has_been_imported(conn, filename, file_hash):
    """Check if the file with this hash was already imported."""
    with conn.cursor() as cur:
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {CSV_HASH_TABLE} (
                filename TEXT PRIMARY KEY,
                file_hash TEXT NOT NULL,
                csv_row_count INTEGER NOT NULL,
                db_row_count INTEGER NOT NULL,
                imported_date TIMESTAMP DEFAULT NOW()
            );
        """
        )
        cur.execute(
            f"""
            SELECT 1 FROM {CSV_HASH_TABLE} WHERE filename = %s
            AND file_hash = %s;
            """,
            (filename, file_hash)
        )
        return cur.fetchone() is not None


def log_import(conn, filename, file_hash, csv_rows, db_rows):
    """Insert or update import log for the file."""
    with conn.cursor() as cur:
        cur.execute(
            f"""
            INSERT INTO {CSV_HASH_TABLE} (filename, file_hash, csv_row_count, imported_date)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (filename) DO UPDATE
            SET file_hash = EXCLUDED.file_hash,
                csv_row_count = EXCLUDED.csv_row_count,
                db_row_count = EXCLUDED.db_row_count,
                imported_date = EXCLUDED.imported_date;
        """,
            (filename, file_hash, csv_rows, db_rows, datetime.now()),
        )
    conn.commit()


def import_csv(conn, csv_file, table_name):
    """Import CSV file into the specified PostgreSQL table."""
    with conn.cursor() as cur, open(csv_file, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        placeholders = ", ".join(["%s"] * len(header))
        for row in reader:
            cur.execute(f"INSERT INTO {table_name} VALUES ({placeholders});", row)
    conn.commit()
    print(f"✅ Imported data from {csv_file} into {table_name}")


def main():
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        connect_timeout=DB_CONNECT_TIMEOUT,
    )

    for csv_path, table_name in CSV_TABLE_MAP.items():
        print(f"\n Processing {csv_path} → {table_name}")
        file_hash = hash_csv(csv_path)
        csv_rows = get_csv_row_count(csv_path)

        if has_been_imported(conn, csv_path, file_hash) is True:
            print(f"Already imported this version of {csv_path}.")

        else:
            db_rows_before = get_db_row_count(conn, table_name)
            print(f"DB rows before import: {db_rows_before}")

            # Write CSV riles to the database
            import_csv(conn, csv_path, table_name)

            # Fetch and save table metadata
            db_rows_after = get_db_row_count(conn, table_name)
            log_import(conn, csv_path, file_hash, csv_rows, db_rows_after)
            print(
                "Imported logged: {csv_path} ({csv_rows} rows) → {table_name} ({db_rows_after} rows)"
            )

    conn.close()


if __name__ == "__main__":
    main()
