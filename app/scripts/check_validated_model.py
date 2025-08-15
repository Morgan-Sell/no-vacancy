"""
Simple script to check if there's a validate model in MLflow staging.
Returns model version via stdout, exists with code 1 if none found.
"""

import sys
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.mlflow_utils import MLflowArtifactLoader


def main():
    """Check for latest validated model  in Staging."""
    try:
        loader = MLflowArtifactLoader()
        model_version = loader.get_model_by_alias("staging")

        if not model_version:
            print("No validated models found in Staging", file=sys.stderr)
            sys.exit(1)

        # Output ONLY the version number to stdout
        print(model_version.version)
        sys.exit(0)

    except Exception as e:
        print(f"Error checking validated model: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
