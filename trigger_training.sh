#!/bin/bash
# Shell script to trigger the training process using Docker Compose
# Script is executed in scheduling_training.yaml as part of the CD pipeline (via GitHub Actions workflow)


set -e # Exit on any error

echo "Starting training workflow..."

# Check if docker-compose is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed"
    exit 1
fi

if ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not available"
    exit 1
fi

# Start infrastructure services (databases, mlflow, etc.)
echo "Starting infrastructure services..."
docker compose up -d bronze-db silver-db gold-db mlflow-db mlflow test-db

# Wait for services to be ready
echo "Waiting for services to be ready..."
for service in bronze-db silver-db gold-db mlflow-db mlflow test-db; do
    echo "Waiting for $service..."
    for i in {1..30}; do
        if docker compose ps $service | grep -qE "(running|Up)"; then
            echo "✅ $service is ready"
            break
        fi
        if [ $i -eq 30 ]; then
            echo "❌ $service failed to start"
            docker compose logs $service
            exit 1
        fi
        sleep 2
    done
done

# Execute training using TrainingOrchestrator class (to be replaced by Airflow in the future)
echo "Executing training via training orchestrator..."
if docker compose --profile training run --rm training-container; then
    echo "✅ Training completed successfully"

    # Cleanup training-specific resources
    echo "Cleaning up training resources..."
    docker compose stop test-db
    docker compose rm -f test-db

    echo "Training workflow completed. Check MLflow for new model artifacts."
    exit 0
else
    echo "❌ Training failed"

    # Show logs for debugging
    echo "Container logs:"
    docker compose logs training-container || true

    # Cleanup
    docker compose down -v --remove-orphans || true
    exit 1
fi
