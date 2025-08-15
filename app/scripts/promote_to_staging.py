import sys
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.mlflow_utils import MLflowArtifactLoader


def main():
    model_version = sys.argv[1]

    loader = MLflowArtifactLoader()

    loader.promote_to_staging(model_version)

    print(f"✅ Model {model_version} promoted to Staging")
    print("📋 Data scientist can now manually approve in MLflow UI")


if __name__ == "__main__":
    main()
