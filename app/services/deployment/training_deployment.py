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
        Deploy by triggering training in training container.
        Used for scheduled retraining or data drift scenarios.
        """
        # Validate environment
        validation_result = self._validate_docker_environment()
        if validation_result:
            return validation_result

        # Execute training
        return self._execute_training_container()

    def _validate_docker_environment(self) -> Optional[dict]:
        """Validate Docker and compose environment. Returns error dict if validation fails."""
        try:
            # Basic environment checks
            self._log_environment_info()

            # Validate docker-compose file
            if not self._validate_compose_file():
                return {
                    "status": "failed",
                    "error": "Docker compose file validation failed",
                }

            # Check if training container exists
            if not self._check_training_container_exists():
                return {
                    "status": "failed",
                    "error": "training-container service not found in docker-compose.yml",
                    "available_services": self._get_available_services(),
                }

            return None  # No errors

        except Exception as e:
            return {
                "status": "failed",
                "error": f"Environment validation failed: {str(e)}",
            }

    def _execute_training_container(self) -> dict:
        """Execute the training container and return results."""
        try:
            cmd = DOCKER_COMPOSE_TRAINING_CMD + [TRAINING_CONTAINER]
            print(f"🚀 Executing: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=TRAINING_DEPLOYMENT_TIMEOUT,
                check=False,
            )

            return self._process_training_result(result)

        except subprocess.TimeoutExpired as e:
            return {
                "status": "failed",
                "error": f"Training timed out after {TRAINING_DEPLOYMENT_TIMEOUT} seconds",
                "stdout": getattr(e, "stdout", ""),
                "stderr": getattr(e, "stderr", ""),
            }
        except Exception as e:
            return {
                "status": "failed",
                "error": f"Unexpected error: {str(e)}",
            }

    def _log_environment_info(self) -> None:
        """Log basic environment information for debugging."""
        import os

        print(f"🔍 Working directory: {os.getcwd()}")
        print(f"🔍 Docker compose file exists: {os.path.exists('docker-compose.yml')}")

        # Get Docker version
        try:
            docker_version = subprocess.run(
                ["docker", "compose", "version"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            print(f"🔍 Docker Compose version: {docker_version.stdout.strip()}")
        except Exception:
            print("⚠️ Could not get Docker Compose version")

    def _validate_compose_file(self) -> bool:
        """Validate docker-compose file syntax."""
        try:
            result = subprocess.run(
                ["docker", "compose", "config", "--quiet"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            is_valid = result.returncode == 0
            print(f"✅ Docker compose config valid: {is_valid}")
            return is_valid
        except Exception:
            print("❌ Docker compose config validation failed")
            return False

    def _check_training_container_exists(self) -> bool:
        """Check if training-container service exists in compose file."""
        try:
            # Check with training profile
            result = subprocess.run(
                ["docker", "compose", "--profile", "training", "config", "--services"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )

            services = result.stdout.strip().split("\n")
            exists = TRAINING_CONTAINER in services
            print(f"🔍 Training container exists: {exists}")
            print(f"🔍 Available services with training profile: {services}")

            return exists
        except Exception as e:
            print(f"❌ Could not check training container: {e}")
            return False

    def _get_available_services(self) -> list:
        """Get list of available services."""
        try:
            result = subprocess.run(
                ["docker", "compose", "--profile", "training", "config", "--services"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            return result.stdout.strip().split("\n") if result.stdout.strip() else []
        except Exception:
            return []

    def _process_training_result(self, result: subprocess.CompletedProcess) -> dict:
        """Process the result of training container execution."""
        print(f"🔍 Training exit code: {result.returncode}")

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

        return {
            "status": "failed",
            "error": f"Training container exited with code {result.returncode}",
            "action": "training_failed",
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode,
        }
