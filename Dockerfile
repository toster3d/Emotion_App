# Use Python 3.10 slim as base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libsndfile1 \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js for the frontend
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv for Python package management
RUN python -m pip install pip --upgrade && \
    python -m pip install uv

# Copy Python requirements file
COPY pyproject.toml .

# Install Python dependencies using uv
RUN uv pip install -e .

# Copy the frontend directory
COPY app/frontend/package.json app/frontend/package-lock.json* app/frontend/

# Install frontend dependencies
WORKDIR /app/app/frontend
RUN npm install

# Copy the frontend source
COPY app/frontend/ ./

# Build the frontend
RUN npm run build

# Copy the rest of the application
WORKDIR /app
COPY app/ app/

# Create directory for model weights
RUN mkdir -p app/models/saved_models

# Run the FastAPI app with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 