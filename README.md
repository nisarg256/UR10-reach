# UR10e Reachability Analysis

This project analyzes and compares the reachable workspace of UR10e robots in different mounting configurations when maintaining end-effector perpendicularity to a wall. This analysis is particularly useful for applications like drywall finishing or wall painting where the tool must remain perpendicular to the work surface.

## Project Overview

The project simulates two UR10e robot configurations:
1. **Flat (Horizontal) Mounting** - Traditional mounting with the robot base parallel to the floor
2. **Perpendicular (Vertical) Mounting** - Wall mounting with the robot base perpendicular to the floor

For each configuration, the project:
- Determines which points on a wall the robot can reach
- Enforces that the tool remains perpendicular to the wall at all times
- Visualizes the reachable workspace
- Compares the reachability of the two configurations

## Prerequisites

- Python 3.7+
- [MuJoCo](https://github.com/deepmind/mujoco) 2.1.0+
- NumPy
- Matplotlib
- SciPy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/UR10-reach.git
cd UR10-reach
```

2. Set up a virtual environment and install dependencies (automatically handled by the scripts):
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## File Structure

### Key Files and Directories

- **src/models/**
  - **ur10e/**: UR10e robot model files
  - **reach_comparison.xml**: MuJoCo XML model with both robot configurations
  
- **src/simulation/**
  - **ik_solver.py**: Inverse kinematics solver with perpendicularity constraints
  - **reach_analyzer.py**: Core reachability analysis implementation
  - **run_reach_analysis.py**: Script to run reachability analysis
  - **simple_viewer.py**: Simple MuJoCo viewer for model visualization
  
- **Shell Scripts**
  - **view_model.sh**: Script to visualize the robot model
  - **analyze_reach.sh**: Script to run reachability analysis
  
- **Results**
  - Results are stored in the `results/` directory (created during analysis)

## Usage

### Model Visualization

To view the model without running any analysis:

```bash
./view_model.sh
```

### Basic Reachability Analysis

To run a basic reachability analysis with default settings:

```bash
./analyze_reach.sh
```

### Advanced Analysis Options

```bash
# Analyze with higher resolution grid
./analyze_reach.sh --y-steps 30 --z-steps 30

# Analyze only the flat-mounted robot
./analyze_reach.sh --robot flat

# Analyze only the perpendicular-mounted robot
./analyze_reach.sh --robot perp

# Demonstrate reachable points by moving the robot
./analyze_reach.sh --demo-boundary

# Save plots to results directory
./analyze_reach.sh --save-plots

# Run in non-interactive batch mode
./analyze_reach.sh --non-interactive --save-plots
```

### Full Options List

```
Usage: ./analyze_reach.sh [options]
Options:
  --model <path>          Path to MuJoCo model XML file
  --non-interactive       Run in non-interactive (batch) mode
  --no-visualization      Disable result visualization
  --demo-boundary         Demonstrate the reachable boundary by moving the robot
  --save-plots            Save the plots to the 'results' directory
  --y-steps <number>      Number of sampling steps along Y axis (default: 15)
  --z-steps <number>      Number of sampling steps along Z axis (default: 15)
  --robot <type>          Which robot to analyze (flat, perp, or both)
  --help                  Show this help message
```

## Visualization Controls

While in the interactive MuJoCo viewer:
- **ESC**: Exit viewer
- **R**: Reset camera view
- **Left mouse drag**: Rotate camera
- **Right mouse drag**: Pan camera (horizontal/vertical)
- **Shift + Right mouse drag**: Pan camera (horizontal/depth)
- **Mouse wheel**: Zoom camera

## Interpreting Results

The analysis generates two types of visualizations:

1. **Individual Reach Maps**: Shows the reachable points for each robot configuration
   - Red points: Reachable by flat-mounted robot
   - Blue points: Reachable by perpendicular-mounted robot

2. **Comparison Map**: Shows the combined reachability
   - Red areas: Reachable only by flat-mounted robot
   - Blue areas: Reachable only by perpendicular-mounted robot
   - Purple areas: Reachable by both configurations

The analysis also prints statistics on the percentage of points reachable by each configuration.

## Results Summary

Based on the most recent analysis (with a 20x20 grid):
- Flat-mounted robot: Reaches 27.5% of the wall area
- Perpendicular-mounted robot: Reaches 37.0% of the wall area

The perpendicular mounting configuration generally provides better overall wall coverage, while the flat mounting provides better coverage near the floor.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.