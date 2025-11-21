#!/bin/bash
# Shell script to trigger the training process using Docker Compose
# Script is executed in scheduling_training.yaml as part of the CD pipeline (via GitHub Actions workflow)


set -e # Exit on any error

echo "Starting training workflow..."

# Check docker availability
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed"
    exit 1
fi

if ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not available"
    exit 1
fi

# Start infrastructure services
echo "Starting infrastructure services..."
if ! docker compose up -d bronze-db silver-db gold-db mlflow-db mlflow test-db; then
    echo "❌ Docker compose up failed"
    echo "=== Container Status ==="
    docker compose ps -a
    echo "=== Container Logs ==="
    docker compose logs --tail=100
    exit 1
fi

# Immediately check for crashed containers
echo "Checking for crashed containers..."
sleep 5  # Give containers a moment to crash if they're going to

CRASHED=$(docker compose ps -a --format json | jq -r 'select(.State == "exited" or .State == "dead") | .Name')
if [ ! -z "$CRASHED" ]; then
    echo "❌ Crashed containers detected:"
    echo "$CRASHED"
    echo ""
    echo "=== Full Status ==="
    docker compose ps -a
    echo ""
    echo "=== Logs from crashed containers ==="
    docker compose logs --tail=50
    exit 1
fi

# Wait for healthy services
echo "Waiting for services to be healthy..."
for service in bronze-db silver-db gold-db mlflow-db mlflow test-db; do
    echo "Waiting for $service..."
    for i in {1..30}; do
        # Check if container exists and is running
        STATUS=$(docker compose ps $service --format json 2>/dev/null | jq -r '.State' || echo "missing")

        if [ "$STATUS" = "running" ]; then
            echo "✅ $service is running"
            break
        elif [ "$STATUS" = "exited" ]; then
            echo "❌ $service exited unexpectedly"
            docker compose logs --tail=50 $service
            exit 1
        fi

        if [ $i -eq 30 ]; then
            echo "❌ $service timeout"
            docker compose ps -a
            docker compose logs $service
            exit 1
        fi
        sleep 2
    done
done

# Execute training
echo "Executing training via training orchestrator..."
if docker compose --profile training run --rm training-container; then
    echo "✅ Training completed successfully"
    docker compose stop test-db
    docker compose rm -f test-db
    echo "Training workflow completed."
    exit 0
else
    echo "❌ Training failed"
    docker compose logs training-container || true
    docker compose down -v --remove-orphans || true
    exit 1
fi
