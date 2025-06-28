import sys

from config import CDConfig
from services.cd_pipeline import CDPipeline


def main():
    """Deploy latest validated model with container restart"""
    # Dependency injection - configuration for deployment
    config = CDConfig.for_production_inference()

    # Initialize pipeline with container deployment strategy
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
