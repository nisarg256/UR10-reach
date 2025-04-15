import os
from pathlib import Path

# Project root directory
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Directories
MODELS_DIR = ROOT_DIR / "src" / "models"
SIMULATION_DIR = ROOT_DIR / "src" / "simulation" 
ANALYSIS_DIR = ROOT_DIR / "src" / "analysis"
UTILS_DIR = ROOT_DIR / "src" / "utils"
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"

# Create directories if they don't exist
for directory in [DATA_DIR, RESULTS_DIR]:
    directory.mkdir(exist_ok=True)

# Simulation parameters
SIM_TIMESTEP = 0.002  # Simulation timestep in seconds
GRAVITY = [0, 0, -9.81]  # Gravity vector

# Robot parameters
UR10_JOINT_LIMITS = {
    "shoulder_pan": [-3.14159, 3.14159],
    "shoulder_lift": [-3.14159, 0],
    "elbow": [0, 3.14159],
    "wrist_1": [-3.14159, 3.14159],
    "wrist_2": [-3.14159, 3.14159],
    "wrist_3": [-3.14159, 3.14159]
}

# Wall dimensions and position
WALL_HEIGHT = 3.0  # meters
WALL_WIDTH = 4.0   # meters
WALL_POSITION = [1.0, 0, 0]  # The wall is 1 meter in front of the robot base

# Mounting configurations
CONFIGURATIONS = {
    "flat": {
        "orientation": [1, 0, 0, 0],  # Quaternion for flat mounting (no rotation)
        "position": [0, 0, 0.1]        # Base positioned slightly above the ground
    },
    "perpendicular": {
        "orientation": [0.7071068, 0, 0.7071068, 0],  # Quaternion for 90-degree rotation around Y
        "position": [0, 0, 0.1]                       # Base positioned slightly above the ground
    }
} 