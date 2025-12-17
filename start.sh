#!/bin/bash

# Render-only deployment: Just run the API server
# Streamlit frontend is deployed separately on Streamlit Community Cloud

PORT=${PORT:-8000}

echo "Starting API server on port $PORT..."
cd /app/backend
exec uvicorn api_server:app --host 0.0.0.0 --port $PORT
