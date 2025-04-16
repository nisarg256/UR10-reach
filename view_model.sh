#!/bin/bash

# Check if Python environment is set up
if [ ! -d "venv" ]; then
    echo "Setting up Python environment..."
    python3 -m venv venv
    . venv/bin/activate
    pip install -r requirements.txt
else
    . venv/bin/activate
fi

# Run the simple MuJoCo viewer with animation
echo "Starting MuJoCo viewer with animation..."
python src/simulation/simple_viewer.py --animate $@ 