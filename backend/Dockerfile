# Use slim Python image with multi-stage build for smaller size
FROM python:3.10-slim AS builder

WORKDIR /app

# Install only essential build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN /usr/local/bin/uv pip install --no-cache --system -r requirements.txt

# --- Final stage ---
FROM python:3.10-slim

WORKDIR /app

# Install only runtime dependencies (git for cloning repos)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy installed packages from builder (uv --system installs to site-packages)
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Make startup script executable
RUN chmod +x start.sh

# Expose API port
EXPOSE 7860

# Run API server
CMD ["./start.sh"]
