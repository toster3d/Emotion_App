# Development stage
FROM python:3.13-slim AS development

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

# Install uv for faster Python package management
RUN python3 -m pip install --no-cache-dir pip --upgrade && \
    python3 -m pip install --no-cache-dir uv

# Copy only requirements file first
COPY pyproject.toml .

# Install Python dependencies using uv
RUN uv pip install -e . --system

# Copy the application code separately (better layer caching)
COPY app/ app/

# Run the FastAPI app with Uvicorn in reload mode for development
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Production stage
FROM python:3.13-slim AS production

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
    libsndfile1 \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster Python package management
RUN python3 -m pip install --no-cache-dir pip --upgrade && \
    python3 -m pip install --no-cache-dir uv

# Copy only requirements file first
COPY pyproject.toml .

# Install Python dependencies using uv
RUN uv pip install -e . --system

# Copy the application code separately (better layer caching)
COPY app/ app/

# Run the FastAPI app with Uvicorn with multiple workers for production
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"] 