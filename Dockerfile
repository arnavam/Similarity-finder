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
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Make startup script executable
RUN chmod +x start.sh

# Expose both ports
EXPOSE 8000 8501

# Use startup script to run both services
CMD ["./start.sh"]
