"""
Utility script to restart inference container only.
Can be called independently or as part of CD pipeline.
"""

import subprocess
import sys

from config import (
    DOCKER_COMPOSE_RESTART_CMD,
    INFERENCE_CONTAINER,
    INFERENCE_DEPLOYMENT_TIMEOUT,
)


def restart_inference_container():
    """Restart only the inference container to load latest model."""
    try:
        print("Restarting inference container...")
        result = subprocess.run(
            [DOCKER_COMPOSE_RESTART_CMD + [INFERENCE_CONTAINER]],
            capture_output=True,
            text=True,
            timeout=INFERENCE_DEPLOYMENT_TIMEOUT,
            check=True,
        )

        if result.returncode == 0:
            print("✅ Inference container restarted successfully")
            return True
        else:
            print(f"❌ Failed to restart container: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("❌ Container restart timed out")
        return False
    except Exception as e:
        print(f"❌ Error restarting container: {e}")
        return False


if __name__ == "__main__":
    success = restart_inference_container()
    sys.exit(0 if success else 1)
