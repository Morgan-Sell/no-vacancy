#!/bin/bash
# Creates test databases for unit and integration tests
set -e

echo "Starting test DBs..."
docker compose --profile test up -d

# 
for db in bronze silver gold; do
    echo "Applying migrations to $db..."
    ALEMBIC_TARGET_DB=$db alembic upgrade head
    echo "Migrations applied to $db"
done

echo "Running tests..."
pytest tests/test_db/ -v -s

echo "Shutting down test DBs..."
docker compose --profile test down -v