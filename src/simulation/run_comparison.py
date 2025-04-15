import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

from src.utils.analysis import compute_reach_on_wall, visualize_workspace, compare_configurations
from src.simulation.simulator import UR10Simulator
from src.config import WALL_POSITION, WALL_HEIGHT, WALL_WIDTH, DATA_DIR, RESULTS_DIR

def run_simulation(config_name, n_samples=5000):
    """
    Run the simulation for a specific configuration.
    
    Args:
        config_name: Name of the configuration ("flat" or "perpendicular")
        n_samples: Number of samples for workspace calculation
        
    Returns:
        Dictionary with simulation results
    """
    print(f"Running simulation for {config_name} configuration...")
    
    # Path to the UR10 URDF file
    model_path = Path(__file__).parent.parent / "models" / "ur10e" / "ur10e.xml"
    
    # Initialize the simulator
    simulator = UR10Simulator(model_path, mounting_config=config_name)
    
    # Compute the workspace
    workspace_data = simulator.compute_workspace(n_samples=n_samples)
    
    # Compute reach on wall
    wall_data = compute_reach_on_wall(
        workspace_data["workspace_points"],
        workspace_data["orientations"],
        WALL_POSITION,
        WALL_HEIGHT,
        WALL_WIDTH
    )
    
    # Combine data
    results = {**workspace_data, **wall_data}
    
    print(f"Simulation for {config_name} configuration completed.")
    print(f"Wall coverage: {wall_data['wall_coverage']:.2%}")
    print(f"Perpendicular points: {len(wall_data['perpendicular_points'])}")
    print(f"Coverage area: {wall_data['coverage_area']:.3f} m²")
    
    return results

def save_results(flat_results, perpendicular_results):
    """
    Save the simulation results.
    
    Args:
        flat_results: Results from flat configuration
        perpendicular_results: Results from perpendicular configuration
    """
    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Save workspace points
    np.save(RESULTS_DIR / "flat_workspace_points.npy", flat_results["workspace_points"])
    np.save(RESULTS_DIR / "perpendicular_workspace_points.npy", perpendicular_results["workspace_points"])
    
    # Save perpendicular points
    np.save(RESULTS_DIR / "flat_perpendicular_points.npy", flat_results["perpendicular_points"])
    np.save(RESULTS_DIR / "perpendicular_perpendicular_points.npy", perpendicular_results["perpendicular_points"])
    
    # Generate and save visualizations
    flat_fig = visualize_workspace(
        flat_results["workspace_points"],
        flat_results["perpendicular_points"],
        WALL_POSITION,
        WALL_HEIGHT,
        WALL_WIDTH,
        title="Flat Mounting Configuration"
    )
    flat_fig.savefig(RESULTS_DIR / "flat_workspace.png")
    
    perpendicular_fig = visualize_workspace(
        perpendicular_results["workspace_points"],
        perpendicular_results["perpendicular_points"],
        WALL_POSITION,
        WALL_HEIGHT,
        WALL_WIDTH,
        title="Perpendicular Mounting Configuration"
    )
    perpendicular_fig.savefig(RESULTS_DIR / "perpendicular_workspace.png")
    
    # Generate comparison report
    comparison = compare_configurations(flat_results, perpendicular_results)
    
    # Save comparison report
    with open(RESULTS_DIR / "comparison_report.txt", "w") as f:
        f.write("# UR10 Mounting Configuration Comparison\n\n")
        f.write(f"Wall position: {WALL_POSITION}\n")
        f.write(f"Wall height: {WALL_HEIGHT} m\n")
        f.write(f"Wall width: {WALL_WIDTH} m\n\n")
        
        f.write("## Flat Configuration\n")
        f.write(f"Workspace volume: {comparison['flat_workspace_volume']:.3f} m³\n")
        f.write(f"Wall coverage: {comparison['flat_wall_coverage']:.2%}\n")
        f.write(f"Perpendicular coverage: {comparison['flat_perpendicular_coverage']:.2%}\n")
        f.write(f"Coverage area: {comparison['flat_coverage_area']:.3f} m²\n\n")
        
        f.write("## Perpendicular Configuration\n")
        f.write(f"Workspace volume: {comparison['perpendicular_workspace_volume']:.3f} m³\n")
        f.write(f"Wall coverage: {comparison['perpendicular_wall_coverage']:.2%}\n")
        f.write(f"Perpendicular coverage: {comparison['perpendicular_perpendicular_coverage']:.2%}\n")
        f.write(f"Coverage area: {comparison['perpendicular_coverage_area']:.3f} m²\n\n")
        
        f.write("## Conclusion\n")
        f.write(f"Better configuration for wall coverage: {comparison['better_configuration'].upper()}\n")
    
    print(f"Results saved to {RESULTS_DIR}")

def main():
    parser = argparse.ArgumentParser(description="Compare UR10 mounting configurations")
    parser.add_argument("--samples", type=int, default=5000, help="Number of samples for workspace calculation")
    args = parser.parse_args()
    
    # Run simulations
    flat_results = run_simulation("flat", n_samples=args.samples)
    perpendicular_results = run_simulation("perpendicular", n_samples=args.samples)
    
    # Save results
    save_results(flat_results, perpendicular_results)
    
    # Print conclusion
    comparison = compare_configurations(flat_results, perpendicular_results)
    print("\nConclusion:")
    print(f"Better configuration for wall coverage: {comparison['better_configuration'].upper()}")

if __name__ == "__main__":
    main() 