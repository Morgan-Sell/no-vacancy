import sys
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CDConfig
from services.cd_pipeline import CDPipeline


def main():
    """Trigger training in training workflow"""
    # Configuration for training workflow
    config = CDConfig.for_automated_training()

    # Initialize pipeline with trianing deployment strategy
    pipeline = CDPipeline(config)

    # Execute training
    result = pipeline.deploy_latest_model()
    print(result)

    # Exit with appropriate code
    if result.startswith("✅"):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
