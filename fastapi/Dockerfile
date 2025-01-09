# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set environment variables to prevent Python from writing .pyc files and to buffer stdout and stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for sentence-transformers and PDF processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker cache
COPY fastapi/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI application code
COPY fastapi /app/fastapi

# Copy the OpenSearch module
COPY opensearch /app/opensearch

# Set the PYTHONPATH to include /app so that FastAPI can find the opensearch module
ENV PYTHONPATH=/app

# Expose the port FastAPI is running on
EXPOSE 8000

# Command to run the FastAPI server with auto-reload for development
CMD ["uvicorn", "fastapi.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
