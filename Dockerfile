# Dependency build stage
FROM python:3.13-slim AS builder

WORKDIR /app

# Installation of system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Installation of package management tools
RUN pip install --upgrade pip uv

# Copying dependency files
COPY pyproject.toml ./
COPY requirements.txt ./

# Installation of dependencies in a separate directory (better cache)
RUN uv pip install -r requirements.txt --system

# Production stage
FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY app/ app/

RUN useradd -m appuser
USER appuser

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
