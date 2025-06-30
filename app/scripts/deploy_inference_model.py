import sys
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CDConfig
from services.cd_pipeline import CDPipeline


def main():
    """Deploy latest validated model with inference container restart."""
    # Dependency injection - inference container configuration
    config = CDConfig.for_production_inference()

    # Initialize pipeline with inference deployment strategy
    pipeline = CDPipeline(config)

    # Execute deployment, including inference container restart
    result = pipeline.deploy_latest_model()
    print(result)

    # Exit with appropriate code for CI/CD
    if result.startswith("✅"):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
