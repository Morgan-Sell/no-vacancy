# Dockerfile (keep this as-is, but add a comment)
# This is the all-in-one development container
# For production, use Dockerfile.training and Dockerfile.prediction

# Python image
FROM python:3.12-slim


# Set the working directory
WORKDIR /app

# Set package directory
ENV PYTHONPATH="/app"

# Install system dependencies, including git, for MLflow
# MLflow requires the git binary to log the Git SHA
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Copy dependencies
COPY requirements.txt .
COPY pyproject.toml .

# Install Python dependencies and clean stale cache
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    find /app -name "__pycache__" -exec rm -rf {} + || true

# Copy the application code
COPY app /app

# Expose API port
EXPOSE 8000

# Default to prediction service, but can be overridden
# Start FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
