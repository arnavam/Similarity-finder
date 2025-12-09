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

# Expose both ports
EXPOSE 8000 8501

CMD ["sh", "-c", "uvicorn api_server:app --host 0.0.0.0 --port 8000 & streamlit run a_streamlit_app.py --server.port=8501 --server.address=0.0.0.0"]
