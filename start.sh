#!/bin/bash
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

PORT=${PORT:-8000}

echo "Starting API server on port $PORT..."
exec uvicorn api_server:app --host 0.0.0.0 --port $PORT
