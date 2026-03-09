#!/bin/zsh
# Startup script for Bundesliga Predictor backend
# Sets DYLD_LIBRARY_PATH so XGBoost can find libomp.dylib

export DYLD_LIBRARY_PATH=/Users/tim/micromamba/lib:$DYLD_LIBRARY_PATH

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
source venv/bin/activate

case "$1" in
  train)
    echo "=== Starting training pipeline ==="
    python -m scripts.train
    ;;
  serve)
    echo "=== Starting API server on http://localhost:8000 ==="
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
    ;;
  *)
    echo "Usage: ./run.sh [train|serve]"
    echo ""
    echo "  train  — ingest data from OpenLigaDB and train models"
    echo "  serve  — start the FastAPI server"
    ;;
esac
