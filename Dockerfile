# Use slim Python image with multi-stage build for smaller size
FROM python:3.10-slim AS builder

WORKDIR /app

# Install only essential build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements-docker.txt .
RUN pip install --no-cache-dir --user -r requirements-docker.txt

# --- Final stage ---
FROM python:3.10-slim

WORKDIR /app

# Install only runtime dependencies (git for cloning repos)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code (API only - Streamlit runs separately)
COPY backend/ ./backend/
COPY test/ ./test/
COPY start.sh .

# Make startup script executable
RUN chmod +x start.sh

# Expose API port
EXPOSE 7860

# Run API server
CMD ["./start.sh"]
