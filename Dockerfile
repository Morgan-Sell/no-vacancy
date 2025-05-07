# Python image
FROM python:3.10-slim


# Set the working directory
WORKDIR /app

# Set package directory
ENV PYTHONPATH="/app"

# Copy dependencies
COPY requirements.txt .
COPY pyproject.toml .

# Install dependencies and clean stale cache
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    find /app -name "__pycache__" -exec rm -rf {} + || true

# Copy the application code
COPY app /app

# Expose API port
EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
