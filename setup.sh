#!/bin/bash

# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install torch matplotlib numpy

echo "Environment setup completed successfully!"