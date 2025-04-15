# UR10 Reach Analysis

This project compares the reach capabilities of a UR10 robot arm in different mounting configurations for construction applications such as drywall finishing.

## Project Goal
Evaluate whether a perpendicular (vertical) or flat (horizontal) mounting configuration provides better reach when the end effector must remain perpendicular to the wall.

## Features
- Simulates two UR10 robot arm configurations in MuJoCo 2.3.2
- Analyzes reachable workspace while maintaining end effector perpendicular to wall
- Provides interactive visualization of reachable areas
- Generates heatmap comparison of wall coverage

## Requirements
- Python 3.x
- MuJoCo 2.3.2
- NumPy, Matplotlib, SciPy

## Quick Start
To run the analysis with interactive visualization:
```bash
./run_visualization.sh
```

This will:
1. Set up a Python virtual environment if needed
2. Install required dependencies
3. Run the reach analysis simulation 
4. Show interactive visualization and save results

## Visualization Controls
- **ESC**: Quit the visualization
- **Space**: Toggle demo mode (robots moving through different configurations)
- **R**: Reset camera view
- **A**: Toggle reachable points visualization

## Understanding the Results
The visualization shows:
- **Red points**: Areas reachable by the flat-mounted robot with end effector perpendicular to wall
- **Blue points**: Areas reachable by the perpendicular-mounted robot with end effector perpendicular to wall
- **Purple areas** (in heatmap): Overlapping reach areas

A heatmap visualization is saved in the `results` directory along with a detailed analysis report.

## Advanced Usage
```bash
# Run with more samples for higher accuracy
./run_visualization.sh --samples 10000

# Load previously computed results instead of recomputing
./run_visualization.sh --load

# Specify a different model file
./run_visualization.sh --model path/to/model.xml

# Run without interactive visualization
python src/simulation/visualize_reach.py
```

## Directory Structure
- `src/models/`: Contains UR10 URDF and MuJoCo model files
- `src/simulation/`: Simulation code for both mounting configurations
- `src/analysis/`: Scripts for analyzing and visualizing reach data
- `src/utils/`: Utility functions

## Usage
[Instructions will be added as the project develops]