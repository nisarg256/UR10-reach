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

# Default parameters
MODEL="src/models/reach_comparison.xml"
INTERACTIVE=true
VISUALIZE=true
DEMO_BOUNDARY=false
SAVE_PLOTS=false
Y_STEPS=15
Z_STEPS=15
ROBOT="both"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --non-interactive)
            INTERACTIVE=false
            shift
            ;;
        --no-visualization)
            VISUALIZE=false
            shift
            ;;
        --demo-boundary)
            DEMO_BOUNDARY=true
            shift
            ;;
        --save-plots)
            SAVE_PLOTS=true
            shift
            ;;
        --y-steps)
            Y_STEPS="$2"
            shift 2
            ;;
        --z-steps)
            Z_STEPS="$2"
            shift 2
            ;;
        --robot)
            ROBOT="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --model <path>          Path to MuJoCo model XML file"
            echo "  --non-interactive       Run in non-interactive (batch) mode"
            echo "  --no-visualization      Disable result visualization"
            echo "  --demo-boundary         Demonstrate the reachable boundary by moving the robot"
            echo "  --save-plots            Save the plots to the 'results' directory"
            echo "  --y-steps <number>      Number of sampling steps along Y axis (default: 15)"
            echo "  --z-steps <number>      Number of sampling steps along Z axis (default: 15)"
            echo "  --robot <type>          Which robot to analyze (flat, perp, or both)"
            echo "  --help                  Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build command line arguments for Python script
ARGS="--model $MODEL"

if [ "$INTERACTIVE" = false ]; then
    ARGS="$ARGS --non-interactive"
fi

if [ "$VISUALIZE" = false ]; then
    ARGS="$ARGS --no-visualization"
fi

if [ "$DEMO_BOUNDARY" = true ]; then
    ARGS="$ARGS --demo-boundary"
fi

if [ "$SAVE_PLOTS" = true ]; then
    ARGS="$ARGS --save-plots"
fi

ARGS="$ARGS --y-steps $Y_STEPS --z-steps $Z_STEPS --robot $ROBOT"

# Run the analysis script
echo "Starting reachability analysis..."
python src/simulation/run_reach_analysis.py $ARGS 