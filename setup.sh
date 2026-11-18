#!/bin/bash

# Create virtual environment

if [[ "$(hostname)" == *"bancada"* ]]; then
    virtualenv -p python .venv
else
    python -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install torch torchvision matplotlib numpy pandas Pillow torchxrayvision

echo "Environment setup completed successfully!"