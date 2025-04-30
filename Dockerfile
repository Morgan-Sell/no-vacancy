# Python image
FROM python:3.10-slim


# Set the working directory
WORKDIR /app

# Set package directory
ENV PYTHONPATH="/app"

# Copy dependencies
COPY requirements.txt .
COPY pyproject.toml .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the application code
COPY app /app

# Expose API port
EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
# ENTRYPOINT ["tail", "-f", "/dev/null"]
