import sys
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from services import MLFLOW_EXPERIMENT_NAME
from services.mlflow_utils import MLflowArtifactLoader


def main():
    """Validate model metrics against thresholds."""
    # Confirm that file name and model version are provided
    # sys.argv is a list containing the CLI arguments passed to a Python script.
    if len(sys.argv) < 2:  # noqa: PLR2004
        print(
            "Usage: python validate_model_metrics.py <model_version> [--mock]",
            file=sys.stderr,
        )
        sys.exit(1)

    model_version = sys.argv[1]
    mock_mode = "--mock" in sys.argv

    print(f"Validating model version: {model_version}")
    print(f"Mock mode: {mock_mode}")

    if mock_mode:
        print("Mock validation passed - model meets thresholds.")
        sys.exit(0)

    try:
        loader = MLflowArtifactLoader()
        model_details = loader.client.get_model_version(
            name=MLFLOW_EXPERIMENT_NAME, version=model_version
        )

        run = loader.client.get_run(model_details.run_id)
        metrics = run.data.metrics

        min_auc = 0.85
        test_auc = metrics.get("test_auc", 0)
        val_auc = metrics.get("val_auc", 0)

        print(f"Test AUC: {test_auc}")
        print(f"Validation AUC: {val_auc}")
        print(f"Minimum AUC threshold: {min_auc}")

        if test_auc >= min_auc and val_auc >= min_auc:
            print("✅ Model (validation & test) meets performance thresholds.")
            sys.exit(0)
        else:
            print("❌ Model does not meet performance thresholds.")
            sys.exit(1)

    except Exception as e:
        print(f"Error validating model: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
