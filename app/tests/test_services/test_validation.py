from unittest.mock import MagicMock, patch

import pytest

from services.validation.base import ModelValidator
from services.validation.manual_validator import ManualValidator


class TestModelValidatorBase:
    """Tests for abstract base class."""

    def test_cannot_instantiate_abstract_class(self):
        """ModelValidator cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            ModelValidator()

    def test_concrete_implementation_works(self):
        """Concrete implementations can be instantiated."""

        class ConcreteValidator(ModelValidator):
            def validate(self, model_version: str):
                return {"valid": True}

        validator = ConcreteValidator()
        result = validator.validate("v1")
        assert result == {"valid": True}


class TestManualValidator:
    """Tests for ManualValidator."""

    @patch("services.validation.manual_validator.MLflowArtifactLoader")
    def test_validate_returns_true_when_approved(self, mock_loader_class):
        """Returns True when model is manually validated."""
        mock_loader = MagicMock()
        mock_loader.check_manual_validation_status.return_value = True
        mock_loader_class.return_value = mock_loader

        validator = ManualValidator()
        result = validator.validate("v1")

        assert result is True
        mock_loader.check_manual_validation_status.assert_called_once_with("v1")

    @patch("services.validation.manual_validator.MLflowArtifactLoader")
    def test_validate_returns_false_when_not_approved(self, mock_loader_class):
        """Returns False when model is not validated."""
        mock_loader = MagicMock()
        mock_loader.check_manual_validation_status.return_value = False
        mock_loader_class.return_value = mock_loader

        validator = ManualValidator()
        result = validator.validate("v1")

        assert result is False

    @patch("services.validation.manual_validator.MLflowArtifactLoader")
    def test_validate_returns_false_on_exception(self, mock_loader_class):
        """Returns False (fail-safe) when exception occurs."""
        mock_loader = MagicMock()
        mock_loader.check_manual_validation_status.side_effect = Exception(
            "MLflow down"
        )
        mock_loader_class.return_value = mock_loader

        validator = ManualValidator()
        result = validator.validate("v1")

        assert result is False
