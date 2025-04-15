#!/usr/bin/env python
import os
import sys
import argparse
from pathlib import Path

# Add the project directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    parser = argparse.ArgumentParser(description="UR10 Reach Analysis")
    parser.add_argument(
        "--samples", 
        type=int, 
        default=5000, 
        help="Number of samples for workspace calculation"
    )
    args = parser.parse_args()
    
    # Create data and results directories
    from src.config import DATA_DIR, RESULTS_DIR
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Run the comparison simulation
    from src.simulation.run_comparison import main as run_comparison
    run_comparison()
    
    print("\nSimulation completed successfully!")
    print(f"Results are saved in the {RESULTS_DIR} directory.")

if __name__ == "__main__":
    main() 