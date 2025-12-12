#!/bin/bash
set -e

# Ensure we are running from the script's directory
cd "$(dirname "$0")"

VENV_DIR="venv"
SCRIPT_NAME="neolabs-chat72b.py"

echo "--- NeocloudX Labs Open Chat Launcher ---"

# 1. Setup Virtual Environment
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# 2. Install Dependencies
echo "Checking dependencies..."
pip install --upgrade pip 
# Core requirements derived from your imports
pip install torch torchvision torchaudio \
    gradio \
    diffusers \
    transformers \
    bitsandbytes \
    accelerate \
    sentencepiece \
    protobuf 

# 3. Run the Python Script
if [ -f "$SCRIPT_NAME" ]; then
    echo "Starting $SCRIPT_NAME..."
    python3 "$SCRIPT_NAME"
else
    echo "Error: $SCRIPT_NAME not found!"
    exit 1
fi
