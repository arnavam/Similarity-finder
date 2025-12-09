FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for textract
RUN apt-get update && apt-get install -y \
    git \
    antiword \
    poppler-utils \
    tesseract-ocr \
    libxml2-dev \
    libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy application code
COPY . .

# Make startup script executable
RUN chmod +x start.sh

# Expose API port
EXPOSE 8000

# Run API server
CMD ["./start.sh"]
