# Python image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Set package directory
ENV PYTHONPATH="/app"

# Copy dependencies
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Copy whole project
COPY . /app

# Expose API port
EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
