# UR10 Reach Analysis - Usage Guide

This project compares the reach capabilities of a UR10 robot arm in different mounting configurations (flat vs. perpendicular) for construction applications such as drywall finishing.

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Simulation

To run the simulation with default parameters:
```bash
python main.py
```

To specify the number of samples for workspace calculation:
```bash
python main.py --samples 10000
```

Note: Higher sample counts provide more accurate results but take longer to compute.

## Results

After running the simulation, the results will be saved in the `results/` directory:

- `flat_workspace.png`: Visualization of the flat mounting configuration workspace
- `perpendicular_workspace.png`: Visualization of the perpendicular mounting configuration workspace
- `comparison_report.txt`: Detailed comparison of the two configurations
- Various `.npy` files containing raw data for further analysis

## Customization

You can modify simulation parameters in `src/config.py`:
- Wall position and dimensions
- Robot mounting configurations
- Joint limits
- Simulation parameters

## Troubleshooting

### MuJoCo Errors
- If you encounter MuJoCo errors, ensure you're using MuJoCo 2.3.2
- The model XML has been adjusted to work with MuJoCo 2.3.2 by changing the integrator from "implicitfast" to "implicit"

### Missing Asset Files
- If mesh asset files are missing, they should be located in `src/models/ur10e/assets/`
- Check that all mesh files exist in this directory 