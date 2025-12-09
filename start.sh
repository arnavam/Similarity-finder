#!/bin/bash

# Start API server in background
echo "Starting API server on port 8000..."
uvicorn api_server:app --host 0.0.0.0 --port 8000 &
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
echo "Starting Streamlit on port 8501..."
exec streamlit run a_streamlit_app.py --server.port=8501 --server.address=0.0.0.0
