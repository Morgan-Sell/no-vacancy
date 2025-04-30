import pytest
from db.postgres import BronzeSessionLocal, GoldSessionLocal, SilverSessionLocal
from schemas.bronze import RawData
from schemas.gold import TestResults, TrainResults, ValidationResults
from schemas.silver import TrainData, ValidationTestData
from sqlalchemy import func

# setup_all_dbs() fixture in conftest.py automatically creates the datatabases


def test_bronze_db(setup_test_dbs):

    with BronzeSessionLocal() as session:
        count = session.query(func.count(RawData.booking_id)).scalar()
        assert count > 0, "Bronze DB is empty"


def test_silver_db_split_train_test_counts(setup_test_dbs):

    with SilverSessionLocal() as session:
        train_count = session.query(func.count(TrainData.booking_id)).scalar()
        test_count = session.query(func.count(ValidationTestData.booking_id)).scalar()
        assert train_count > 0, "Silver DB train table is empty"
        assert test_count > 0, "Silver DB validate/test table is empty"


def test_gold_db_predictions_are_binary(setup_test_dbs):

    with GoldSessionLocal() as session:
        results = session.query(TrainResults.prediction).distinct().all()
        unique_preds = {result[0] for result in results}
        assert unique_preds.issubset(
            {0, 1}
        ), "Unexpected prediction values: {unique_preds}"


def test_gold_db_probabilities_sum_to_one(setup_test_dbs):

    with GoldSessionLocal() as session:
        rows = session.query(TrainResults).limit(10).all()
        for row in rows:
            total = round(row.probability_not_canceled + row.probability_canceled, 2)
            assert abs(total - 1.0) < 0.01, f"Probabilities do not sum to 1: {total}"
