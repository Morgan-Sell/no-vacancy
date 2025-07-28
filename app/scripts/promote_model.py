"""
Simple script to promote a model version to Production in MLflow.
Usage: python promote_model.py <model_version>
"""

import sys
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from services.mlflow_utils import MLflowArtifactLoader


def main():
    """Promote specified model version to Production."""
    # Confirm that file name and model version are provided
    # sys.argv is a list containing the CLI arguments passed to a Python script.
    if len(sys.argv) != 2:  # noqa: PLR2004
        print("Usage: python promote_model.py <model_version>", file=sys.stderr)
        sys.exit(1)

    model_version = sys.argv[1]

    try:
        loader = MLflowArtifactLoader()
        loader.promote_to_production(model_version)

        print(f"Successfully promoted model version {model_version} tp Production.")
        sys.exit(0)

    except Exception as e:
        print(f"Error promoting model: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
