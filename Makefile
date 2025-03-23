.PHONY: test-integration

test-integration:
	@echo "🚀 Starting test Postgres container..."
	docker compose up -d test-db
	@echo "⏳ Waiting for Postgres to be ready..."
	sleep 5
	@echo "Running integration tests..."
	POSTGRES_HOST=localhost \
	POSTGRES_PORT=5433 \
	POSTGRES_DB=test_db \
	POSTGRES_USER=test_user \
	POSTGRES_PASSWORD=test_password \
	pytest tests/test_scripts/test_import_csv_integration.py -s -v
	@echo "🛑 Stopping test Postgres container..."
	docker compose down
	@echo "✅ Integration tests completed."