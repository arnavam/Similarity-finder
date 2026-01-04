#!/bin/bash

# Render-only deployment: Just run the API server
# Streamlit frontend is deployed separately on Streamlit Community Cloud

PORT=${PORT:-7860}

echo "Starting API server on port $PORT..."
# We are already in /app
exec uvicorn api_server:app --host 0.0.0.0 --port $PORT --workers 2 --access-log
