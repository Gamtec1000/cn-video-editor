#!/bin/bash
# Start Video Editor services in development mode (without Docker)

set -e

PROJECT_ROOT="$(dirname "$(dirname "$(readlink -f "$0")")")"
VENV_DIR="$PROJECT_ROOT/.venv"

echo "========================================"
echo "CN Video Editor - Development Start"
echo "========================================"

# Create virtual environment if not exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate venv
source "$VENV_DIR/bin/activate"

# Install dependencies
echo "Installing dependencies..."
pip install -q -r "$PROJECT_ROOT/services/ingest/requirements.txt"
pip install -q -r "$PROJECT_ROOT/services/transcribe/requirements.txt"

# Set environment
export MEDIA_BASE="/home/gamtec1000/cn_media"
export WHISPER_MODEL="base"

# Create media directories
mkdir -p "$MEDIA_BASE/raw" "$MEDIA_BASE/thumbnails" "$MEDIA_BASE/audio" "$MEDIA_BASE/transcripts" "$MEDIA_BASE/.jobs"

# Start services
echo ""
echo "Starting services..."
echo "  Ingest:     http://localhost:8001"
echo "  Transcribe: http://localhost:8002"
echo ""

# Run services in background
cd "$PROJECT_ROOT/services/ingest"
python3 main.py &
INGEST_PID=$!

cd "$PROJECT_ROOT/services/transcribe"
python3 main.py &
TRANSCRIBE_PID=$!

# Trap to cleanup on exit
cleanup() {
    echo ""
    echo "Stopping services..."
    kill $INGEST_PID 2>/dev/null || true
    kill $TRANSCRIBE_PID 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM

echo "Services started. Press Ctrl+C to stop."
echo ""

# Wait for background jobs
wait
