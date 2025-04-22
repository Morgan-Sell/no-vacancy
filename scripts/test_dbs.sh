#!/bin/bash
# Creates test databases for unit and integration tests
set -e

echo "Starting test DBs..."
docker compose --profile test up -d

echo "Running tests..."
pytest tests/test_db/ -v -s

echo "Shutting down test DBs..."
docker compose --profile test down -v