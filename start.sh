#!/bin/bash

# Use PORT env variable (Render sets this) or default to 8501
STREAMLIT_PORT=${PORT:-8501}
API_PORT=8000

# Start API server in background
echo "Starting API server on port $API_PORT..."
uvicorn api_server:app --host 0.0.0.0 --port $API_PORT &
API_PID=$!

# Wait for API to be ready
echo "Waiting for API to start..."
sleep 3

# Check if API is running
if ! kill -0 $API_PID 2>/dev/null; then
    echo "ERROR: API server failed to start!"
    exit 1
fi

echo "API server running with PID $API_PID"

# Start Streamlit in foreground (keeps container alive)
echo "Starting Streamlit on port $STREAMLIT_PORT..."
exec streamlit run a_streamlit_app.py --server.port=$STREAMLIT_PORT --server.address=0.0.0.0
