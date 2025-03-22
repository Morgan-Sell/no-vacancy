import hashlib
import os
from scripts.import_csv_to_postgres import hash_csv, get_csv_row_count


def test_hash_csv(temp_booking_data_csv):
    with open(temp_booking_data_csv, "rb") as f:
        expected_hash = hashlib.sha256(f.read()).hexdigest()
    assert hash_csv(temp_booking_data_csv) == expected_hash


def test_get_csv_row_count(temp_booking_data_csv):
    row_count = get_csv_row_count(temp_booking_data_csv)
    assert row_count == 20