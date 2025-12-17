from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from db.db_init import (
    bronze_db,
    silver_db,
    gold_db,
    test_db,
    init_all_databases,
)


class TestDatabaseInstances:
    """Test database instances are configured correctly."""

    def test_bronze_db_exists(self):
        """Bronze database instance is created."""
        assert bronze_db is not None
        assert bronze_db.db_name == "novacancy-bronze"

    def test_silver_db_exists(self):
        """Silver database instance is created."""
        assert silver_db is not None
        assert silver_db.db_name == "novacancy-silver"

    def test_gold_db_exists(self):
        """Gold database instance is created."""
        assert gold_db is not None
        assert gold_db.db_name == "novacancy-gold"

    def test_test_db_exists(self):
        """Test database instance is created."""
        assert test_db is not None
        assert test_db.db_name == "novacancy-test"


class TestInitAllDatabases:
    """Tests for init_all_databases function."""

    @pytest.mark.asyncio
    @patch("db.db_init.test_db")
    @patch("db.db_init.gold_db")
    @patch("db.db_init.silver_db")
    @patch("db.db_init.bronze_db")
    async def test_init_all_databases_creates_tables(
        self, mock_bronze, mock_silver, mock_gold, mock_test
    ):
        """All database tables are created."""
        # Setup mock engines with async context managers
        for mock_db in [mock_bronze, mock_silver, mock_gold, mock_test]:
            mock_engine = MagicMock()
            mock_conn = MagicMock()
            mock_conn.run_sync = AsyncMock()

            # Create async context manager
            mock_context = AsyncMock()
            mock_context.__aenter__.return_value = mock_conn
            mock_context.__aexit__.return_value = None
            mock_engine.begin.return_value = mock_context

            mock_db.create_engine.return_value = mock_engine

        await init_all_databases()

        # Verify each database had create_all called
        mock_bronze.create_engine.assert_called_once()
        mock_silver.create_engine.assert_called_once()
        mock_gold.create_engine.assert_called_once()

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"CONTAINER_TYPE": "training"})
    @patch("db.db_init.test_db")
    @patch("db.db_init.gold_db")
    @patch("db.db_init.silver_db")
    @patch("db.db_init.bronze_db")
    async def test_init_skips_test_db_in_training_container(
        self, mock_bronze, mock_silver, mock_gold, mock_test
    ):
        """Test DB initialization is skipped in training containers."""
        for mock_db in [mock_bronze, mock_silver, mock_gold]:
            mock_engine = MagicMock()
            mock_conn = MagicMock()
            mock_conn.run_sync = AsyncMock()

            mock_context = AsyncMock()
            mock_context.__aenter__.return_value = mock_conn
            mock_context.__aexit__.return_value = None
            mock_engine.begin.return_value = mock_context

            mock_db.create_engine.return_value = mock_engine

        await init_all_databases()

        # test_db should NOT be called when CONTAINER_TYPE=training
        mock_test.create_engine.assert_not_called()

    @pytest.mark.asyncio
    @patch.dict("os.environ", {}, clear=False)
    @patch("db.db_init.test_db")
    @patch("db.db_init.gold_db")
    @patch("db.db_init.silver_db")
    @patch("db.db_init.bronze_db")
    @patch("builtins.print")
    async def test_init_handles_test_db_exception(
        self, mock_print, mock_bronze, mock_silver, mock_gold, mock_test
    ):
        """Exception in test_db initialization is caught and logged."""
        # Remove CONTAINER_TYPE so test_db init is attempted
        import os

        os.environ.pop("CONTAINER_TYPE", None)

        # Setup working mocks for bronze, silver, gold
        for mock_db in [mock_bronze, mock_silver, mock_gold]:
            mock_engine = MagicMock()
            mock_conn = MagicMock()
            mock_conn.run_sync = AsyncMock()

            mock_context = AsyncMock()
            mock_context.__aenter__.return_value = mock_conn
            mock_context.__aexit__.return_value = None
            mock_engine.begin.return_value = mock_context

            mock_db.create_engine.return_value = mock_engine

        # Setup test_db to raise exception
        mock_test_engine = MagicMock()
        mock_test_engine.begin.side_effect = Exception("Connection refused")
        mock_test.create_engine.return_value = mock_test_engine

        # Should not raise - exception is caught
        await init_all_databases()

        # Verify warning was printed
        mock_print.assert_called()
        call_args = str(mock_print.call_args)
        assert "Test DB initialization skipped" in call_args
