import subprocess
from typing import Optional

from config import (
    DOCKER_COMPOSE_TRAINING_CMD,
    TRAINING_CONTAINER,
    TRAINING_DEPLOYMENT_TIMEOUT,
)
from services.deployment.base import DeploymentStrategy


class TrainingContainerDeployment(DeploymentStrategy):
    """
    Deployment strategy that triggers training in dedicated training container.
    Used for automated retraining workflows.
    """

    def deploy(self, model_version: Optional[str] = None) -> dict:
        """
        Deploy by trigering training in training container.
        Used for scheduled retraiing or data drift scenarios.
        """
        try:
            # Use the specific training command that includes the profile
            cmd = DOCKER_COMPOSE_TRAINING_CMD + [TRAINING_CONTAINER]

            print(f"🔍 DEBUG: Executing command: {' '.join(cmd)}")
            print(
                f"🔍 DEBUG: Working directory: {subprocess.run(['pwd'], capture_output=True, text=True, check=False).stdout.strip()}"
            )
            print(
                f"🔍 DEBUG: Docker compose version: {subprocess.run(['docker', 'compose', 'version'], capture_output=True, text=True, check=False).stdout}"
            )

            # Check if docker-compose.yml exists
            import os

            if os.path.exists("docker-compose.yml"):
                print("✅ docker-compose.yml found")
            else:
                print("❌ docker-compose.yml NOT found")
                print(f"📁 Files in current directory: {os.listdir('.')}")

            # Validate the docker-compose configuration
            print("🔍 DEBUG: Validating docker-compose config...")
            validate_result = subprocess.run(
                ["docker", "compose", "config", "--quiet"],
                capture_output=True,
                text=True,
                check=False,
            )
            if validate_result.returncode != 0:
                print(
                    f"❌ Docker compose config validation failed: {validate_result.stderr}"
                )
            else:
                print("✅ Docker compose config is valid")

            # List available services
            print("🔍 DEBUG: Available services:")
            services_result = subprocess.run(
                ["docker", "compose", "config", "--services"],
                capture_output=True,
                text=True,
                check=False,
            )
            print(f"Services: {services_result.stdout}")

            # Check if training-container service exists
            if "training-container" not in services_result.stdout:
                return {
                    "status": "failed",
                    "error": "training-container service not found in docker-compose.yml",
                    "available_services": services_result.stdout.strip().split("\n"),
                }

            # Run training container (it will exit when training completes)
            print("🚀 Starting training container...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=TRAINING_DEPLOYMENT_TIMEOUT,
                check=False,  # Handle return codes manually for better error messages
            )

            print(f"🔍 DEBUG: Return code: {result.returncode}")
            print("🔍 DEBUG: Command completed")

            # Always print stdout and stderr for debugging
            if result.stdout:
                print("📝 STDOUT:")
                print("=" * 50)
                print(result.stdout)
                print("=" * 50)

            if result.stderr:
                print("⚠️ STDERR:")
                print("=" * 50)
                print(result.stderr)
                print("=" * 50)

            if result.returncode == 0:
                return {
                    "status": "success",
                    "action": "training_triggered",
                    "message": "Training completed successfully.",
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            else:
                return {
                    "status": "failed",
                    "error": f"Training container exited with code {result.returncode}",
                    "action": "training_failed",
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "return_code": result.returncode,
                }

        except subprocess.TimeoutExpired as e:
            return {
                "status": "failed",
                "error": f"Training timed out after {TRAINING_DEPLOYMENT_TIMEOUT} seconds",
                "stdout": e.stdout if e.stdout else "",
                "stderr": e.stderr if e.stderr else "",
            }

        except FileNotFoundError as e:
            return {
                "status": "failed",
                "error": f"Docker compose command not found: {e}. Make sure Docker is installed and running.",
            }

        except Exception as e:
            return {
                "status": "failed",
                "error": f"Unexpected error: {str(e)}",
                "exception_type": type(e).__name__,
            }
