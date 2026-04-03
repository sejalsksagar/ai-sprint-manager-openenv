#!/bin/sh
# Start FastAPI env server in background
uvicorn server:app --host 0.0.0.0 --port 8000 &

# Wait for it to be ready
sleep 3

# Start Gradio UI in foreground (HF Spaces expects port 7860)
python ui.py