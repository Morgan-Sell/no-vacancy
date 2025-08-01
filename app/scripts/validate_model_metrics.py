import sys
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from services import MLFLOW_AUC_THRESHOLD, MLFLOW_EXPERIMENT_NAME
from services.mlflow_utils import MLflowArtifactLoader


def validate_model_by_version(loader: MLflowArtifactLoader, model_version: str) -> bool:
    """Validate model metrics by version number."""
    try:
        # Get model details by version
        model_details = loader.client.get_model_version(
            name=MLFLOW_EXPERIMENT_NAME, version=model_version
        )

        # Get run metrics
        run = loader.client.get_run(model_details.run_id)
        metrics = run.data.metrics

        return _check_metrics_thresholds(metrics)

    except Exception as e:
        print(f"Error geting model version {model_version}: {e}", file=sys.stderr)
        return False


def validate_model_by_alias(loader: MLflowArtifactLoader, alias: str) -> bool:
    """Validate model metrics by alias (preferred method)."""
    try:
        # Use the new artifact metadata method
        metadata = loader.get_artifact_metadata_by_alias(alias)

        if not metadata:
            print(f"No model found with alias '{alias}", file=sys.stderr)
            return False

        if metadata.get("partial_data"):
            print(f"Warning: Could not retrieve complete metrics for alias '{alias}")
            return False

        metrics = metadata.get("metrics", {})
        return _check_metrics_thresholds(metrics)

    except Exception as e:
        print(f"Error validating model with alias '{alias}': {e}", file=sys.stderr)
        return False


def _check_metrics_thresholds(metrics: dict) -> bool:
    """Check if metrics meet minimum thresholds."""
    test_auc = metrics.get("test_auc", 0)
    val_auc = metrics.get("val_auc", 0)

    print(f"Test AUC: {test_auc}")
    print(f"Validation AUC: {val_auc}")
    print(f"Minimum AUC threshold: {MLFLOW_AUC_THRESHOLD}")

    if test_auc >= MLFLOW_AUC_THRESHOLD and val_auc >= MLFLOW_AUC_THRESHOLD:
        print("✅ Model (validation & test) meets performance thresholds.")
        return True
    else:
        print("❌ Model does not meet performance thresholds.")
        return False


def main():
    """Validate model metrics against thresholds."""
    # Confirm that file name and model version are provided
    # sys.argv is a list containing the CLI arguments passed to a Python script.
    if len(sys.argv) < 2:  # noqa: PLR2004
        print(
            "Usage: python validate_model_metrics.py <model_version_or_alias> [--mock] [--alias]",
            file=sys.stderr,
        )
        print("Examples:")
        print(
            "  python validate_model_metrics.py 5                    # Validate version 5"
        )
        print(
            "  python validate_model_metrics.py staging --alias      # Validate staging alias"
        )
        print(
            "  python validate_model_metrics.py production --alias   # Validate production alias"
        )
        sys.exit(1)

    model_identifier = sys.argv[1]
    mock_mode = "--mock" in sys.argv
    use_alias = "--alias" in sys.argv

    print(f"Validating model: {model_identifier}")
    print(f"Using alias: {use_alias}")
    print(f"Mock mode: {mock_mode}")

    if mock_mode:
        print("Mock validation passed - model meets thresholds.")
        sys.exit(0)

    try:
        loader = MLflowArtifactLoader()

        # Choose validation method base on flag
        if use_alias:
            validation_passed = validate_model_by_alias(loader, model_identifier)
        else:
            validation_passed = validate_model_by_version(loader, model_identifier)

        # Exit with appropriate code
        sys.exit(0 if validation_passed else 1)

    except Exception as e:
        print(f"Error during validation: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
