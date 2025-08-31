"""
Integration test script for Airflow training orchestration.
Tests the underlying codebase functionality, not the Airflow DAG itself.
"""

import pytest
from unittest.mock import MagicMock, patch
from config import get_logger
from scripts.import_csv_to_postgres import main as import_csv_data
from services.trainer import train_pipeline
from services.predictor import make_prediction

from airflow.dags.training_pipeline_dag import validate_model_artifacts

logger = get_logger(logger_name=__name__)


class TestOrchestrationIntegration:
    """Test the critical orchestration workflow."""

    def test_step_1_csv_import(self, booking_data):
        """Test CSV import to Bronze DB"""
        logger.info("🧪 Testing Step 1: CSV Import to Bronze")

        with (
            patch("scripts.import_csv_to_postgres.import_csv") as mock_import,
            patch("scripts.import_csv_to_postgres.has_been_imported") as mock_check,
            patch("scripts.import_csv_to_postgres.log_import") as mock_log,
        ):

            mock_check.return_value = False  # Simulate new file
            mock_import.return_value = None
            mock_log.return_value = None

            # Actually call the function
            result = import_csv_data()

            # Assert
            mock_import.assert_called_once()
            mock_log.assert_called_once()
            logger.info("✅ CSV import test passed")

    @pytest.mark.asyncio
    async def test_step_2_training(
        self, booking_data, mock_pipeline, mock_processor, mock_mlflow
    ):
        """Test model training workflow using existing fixtures"""
        logger.info("🧪 Testing Step 2: Model Training")

        with (
            patch("services.trainer.load_raw_data") as mock_load,
            patch("services.trainer.build_pipeline") as mock_build,
            patch("services.trainer.save_to_silver_db") as mock_save,
        ):

            mock_load.return_value = booking_data
            mock_build.return_value = mock_pipeline
            mock_save.return_value = None

            # Act
            await train_pipeline()

            # Assert
            mock_load.assert_called_once()
            mock_build.assert_called_once()
            mock_save.assert_called_once()
            mock_pipeline.fit.assert_called_once()

            logger.info("✅ Model training test passed")

    @pytest.mark.asyncio
    async def test_step_3_predictions(
        self, preprocessed_booking_data, mock_mlflow_pipeline, mock_mlflow_processor
    ):
        """Test prediction generation using existing mock fixture"""
        logger.info("🧪 Testing Step 3: Prediction Generation")

        X_processed, y_processed = preprocessed_booking_data
        test_data = X_processed.copy()
        test_data["is_cancellation"] = y_processed

        with patch(
            "services.predictor.load_pipeline_and_processor_from_mlflow"
        ) as mock_load:
            mock_load.return_value = (mock_mlflow_pipeline, mock_mlflow_processor)

            # Act
            result = await make_prediction(test_data, already_processed=True)

            # Assert
            assert result is not None
            assert len(result) == len(test_data)
            assert "prediction" in result.columns
            assert "booking_id" in result.columns

            mock_load.assert_called_once()
            mock_mlflow_pipeline.predict.assert_called_once()
            mock_mlflow_pipeline.predict_proba.assert_called_once()

            logger.info("✅ Prediction generation test passed")

    def test_step_4_validation(self):
        """Test MLflow validation logic"""
        logger.info("🧪 Testing Step 4: MLflow Validation")

        with patch("requests.get") as mock_get:
            from unittest.mock import MagicMock

            mock_versions_response = MagicMock()
            mock_versions_response.status_code = 200
            mock_versions_response.json.return_value = {
                "model_versions": [{"version": "3", "run_id": "test-run-789"}]
            }

            mock_metrics_response = MagicMock()
            mock_metrics_response.status_code = 200
            mock_metrics_response.json.return_value = {
                "run": {"data": {"metrics": {"test_auc": 0.88, "val_auc": 0.86}}}
            }

            mock_get.side_effect = [mock_versions_response, mock_metrics_response]

            # Act
            result = validate_model_artifacts()

            # Assert
            assert result == "Validation passed"

            # Assert API was called twice (versions + metrics)
            assert mock_get.call_count == 2

            logger.info("✅ MLflow validation test passed")

    @pytest.mark.asyncio
    async def test_full_orchestration_flow(
        self,
        booking_data,
        preprocessed_booking_data,
        mock_pipeline,
        mock_processor,
        mock_mlflow_pipeline,
        mock_mlflow_processor,
        mock_mlflow,
    ):
        """Test the complete orchestration flow end-to-end"""
        logger.info("🧪 Testing Full Orchestration Flow")

        # Step 1: CSV Import
        with (
            patch("scripts.import_csv_to_postgres.import_csv") as mock_import,
            patch(
                "scripts.import_csv_to_postgres.has_been_imported", return_value=False
            ),
            patch("scripts.import_csv_to_postgres.log_import"),
        ):

            import_csv_data()
            mock_import.assert_called_once()

        # Step 2: Training
        with (
            patch("services.trainer.load_raw_data", return_value=booking_data),
            patch("services.trainer.build_pipeline", return_value=mock_pipeline),
            patch("services.trainer.save_to_silver_db"),
        ):

            await train_pipeline()
            mock_pipeline.fit.assert_called_once()

        # Step 3: Predictions
        X_processed, y_processed = preprocessed_booking_data
        test_data = X_processed.copy()
        test_data["is_cancellation"] = y_processed

        with patch(
            "services.predictor.load_pipeline_and_processor_from_mlflow",
            return_value=(mock_mlflow_pipeline, mock_mlflow_processor),
        ):
            result = await make_prediction(test_data, already_processed=True)
            assert result is not None
            assert len(result) > 0

        # Step 4: Validation
        with patch("requests.get") as mock_get:
            mock_versions_response = MagicMock()
            mock_versions_response.status_code = 200
            mock_versions_response.json.return_value = {
                "model_versions": [{"version": "3", "run_id": "test-run-789"}]
            }

            mock_metrics_response = MagicMock()
            mock_metrics_response.status_code = 200
            mock_metrics_response.json.return_value = {
                "run": {"data": {"metrics": {"test_auc": 0.88, "val_auc": 0.86}}}
            }

            mock_get.side_effect = [mock_versions_response, mock_metrics_response]

            # Act
            validation_result = validate_model_artifacts()

            # Assert
            assert validation_result == "Validation passed"

        logger.info("🎉 Full orchestration flow test passed!")
