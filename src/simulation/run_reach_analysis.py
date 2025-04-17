#!/usr/bin/env python3
import os
import sys
import time
import argparse
import numpy as np
import mujoco
from pathlib import Path
import matplotlib.pyplot as plt

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.simulation.ik_solver import UR10eIKSolver
from src.simulation.reach_analyzer import ReachabilityAnalyzer

# Default model path
DEFAULT_MODEL = "src/models/reach_comparison.xml"

def setup_visualization(model, data):
    """Set up MuJoCo visualization."""
    from mujoco.glfw import glfw
    
    # Initialize visualization data structures
    scene = mujoco.MjvScene(model, maxgeom=10000)
    camera = mujoco.MjvCamera()
    option = mujoco.MjvOption()
    
    # Initialize GLFW
    if not glfw.init():
        print("Could not initialize GLFW")
        sys.exit(1)
    
    # Create window
    window = glfw.create_window(1200, 900, "UR10 Reachability Analysis", None, None)
    if not window:
        glfw.terminate()
        print("Could not create window")
        sys.exit(1)
    
    # Make context current
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    
    # Initialize MuJoCo renderer
    mujoco.mjv_defaultCamera(camera)
    mujoco.mjv_defaultOption(option)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
    
    # Set camera view
    camera.azimuth = 90
    camera.elevation = -20
    camera.distance = 4.0
    camera.lookat[0] = 0.0
    camera.lookat[1] = 0.0
    camera.lookat[2] = 1.0
    
    # Initialize mouse interaction
    button_left = False
    button_middle = False
    button_right = False
    lastx = 0
    lasty = 0
    mods_shift = False

    # Mouse button callback
    def mouse_button(window, button, act, mods):
        nonlocal button_left, button_middle, button_right, lastx, lasty
        
        # Update button state
        button_left = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
        button_middle = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
        button_right = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)
        
        # Update cursor position
        lastx, lasty = glfw.get_cursor_pos(window)

    # Mouse move callback
    def mouse_move(window, xpos, ypos):
        nonlocal lastx, lasty, button_left, button_middle, button_right, mods_shift
        
        # Compute mouse displacement
        dx = xpos - lastx
        dy = ypos - lasty
        
        # Update camera view with mouse movement
        if button_right:
            # Pan camera
            if mods_shift:
                camera.lookat[0] -= 0.01 * dx
                camera.lookat[1] += 0.01 * dy
            else:
                camera.lookat[0] -= 0.01 * dx
                camera.lookat[2] += 0.01 * dy
        elif button_left:
            # Rotate camera
            camera.azimuth -= 0.3 * dx
            camera.elevation -= 0.3 * dy
            
        # Update cursor position
        lastx = xpos
        lasty = ypos

    # Scroll callback for zooming
    def scroll(window, xoffset, yoffset):
        camera.distance *= 0.9 if yoffset > 0 else 1.1

    # Key callback
    def key_callback(window, key, scancode, action, mods):
        # Reset camera view on 'R' press
        if key == glfw.KEY_R and action == glfw.PRESS:
            mujoco.mjv_defaultCamera(camera)
            camera.azimuth = 90
            camera.elevation = -20
            camera.distance = 4.0
            camera.lookat[0] = 0.0
            camera.lookat[1] = 0.0
            camera.lookat[2] = 1.0
        
        # Exit on ESC
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)

    # Set callbacks
    glfw.set_mouse_button_callback(window, mouse_button)
    glfw.set_cursor_pos_callback(window, mouse_move)
    glfw.set_scroll_callback(window, scroll)
    glfw.set_key_callback(window, key_callback)
    
    return window, scene, camera, option, context


def render_frame(window, model, data, scene, camera, option, context):
    """Render a single frame."""
    from mujoco.glfw import glfw
    
    # Update modifier tracking
    mods_shift = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or \
                glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    
    # Get framebuffer viewport
    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)
    
    # Update scene
    mujoco.mjv_updateScene(
        model, data, option, None, camera,
        mujoco.mjtCatBit.mjCAT_ALL.value, scene
    )
    
    # Render
    mujoco.mjr_render(viewport, scene, context)
    
    # Swap
    glfw.swap_buffers(window)
    glfw.poll_events()
    
    # Check for window close
    return not glfw.window_should_close(window)


def progress_callback(y_idx, z_idx, total_y, total_z):
    """Callback to show progress of reachability analysis."""
    total_points = total_y * total_z
    current_point = y_idx * total_z + z_idx + 1
    progress = current_point / total_points * 100
    
    # Update progress bar
    bar_length = 30
    filled_length = int(bar_length * current_point // total_points)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    
    # Print progress
    print(f"\rProgress: [{bar}] {progress:.1f}% ({current_point}/{total_points})", end='')
    
    if current_point == total_points:
        print()  # Add newline when done


def run_analysis(model_path, interactive=True, visualize=True, demo_boundary=False, 
                 save_plots=False, y_steps=15, z_steps=15, robot_type="both"):
    """
    Run the reachability analysis.
    
    Args:
        model_path: Path to MuJoCo model XML file
        interactive: Whether to run in interactive mode
        visualize: Whether to visualize the results
        demo_boundary: Whether to demonstrate the reachable boundary
        save_plots: Whether to save the plots
        y_steps: Number of steps along Y axis
        z_steps: Number of steps along Z axis
        robot_type: Which robot to analyze ("flat", "perp", or "both")
    """
    # Load model
    print(f"Loading model from {model_path}")
    
    try:
        model = mujoco.MjModel.from_xml_path(str(model_path))
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Reset data to initial state
    mujoco.mj_resetData(model, data)
    
    # Print some model information
    print(f"Model loaded: {model.nq} DOFs, {model.nbody} bodies")
    
    # Initialize reach analyzer
    analyzer = ReachabilityAnalyzer(model, data)
    
    # Set up visualization if needed
    window = None
    scene = None
    camera = None
    option = None
    context = None
    
    if interactive:
        window, scene, camera, option, context = setup_visualization(model, data)
    
    # Set up grid and analyze reachability
    print("Setting up grid for reachability analysis...")
    analyzer.setup_grid(y_steps=y_steps, z_steps=z_steps)
    
    if interactive:
        print("\nPerforming reachability analysis (this may take some time)...")
        analyzer.analyze_reachability(progress_callback=progress_callback, robot_type=robot_type)
        
        # Plot results
        if visualize:
            print("\nGenerating plots...")
            
            # Create figure directory if it doesn't exist
            os.makedirs("results", exist_ok=True)
            
            # Plot individual reachability maps
            reachability_fig = analyzer.plot_reachability_maps()
            if save_plots:
                reachability_fig.savefig("results/reachability_maps.png", dpi=300, bbox_inches='tight')
            
            # Plot comparison
            comparison_fig = analyzer.plot_comparison()
            if save_plots:
                comparison_fig.savefig("results/reachability_comparison.png", dpi=300, bbox_inches='tight')
                print("Plots saved to 'results' directory")
            
            # Make sure the plots are displayed and stay open
            plt.ion()  # Enable interactive mode
            reachability_fig.canvas.draw()
            comparison_fig.canvas.draw()
            plt.pause(0.001)  # Small pause to ensure display processes
        
        # Demonstrate boundary if requested
        if demo_boundary:
            # Create a custom visualization-aware pause function
            def visualized_pause(duration):
                """Pause with visualization updates."""
                start_time = time.time()
                while time.time() - start_time < duration:
                    if not render_frame(window, model, data, scene, camera, option, context):
                        return False  # Terminate if window is closed
                    time.sleep(0.01)  # Small sleep to prevent CPU hogging
                return True
            
            # Create demo function that uses visualization
            def demo_boundary_with_viz(robot_type):
                """Demonstrate boundary with visualization updates."""
                # Get boundary points
                boundary_points = analyzer.find_reachable_boundary(robot_type)
                
                if not boundary_points:
                    print(f"No reachable points found for {robot_type} robot.")
                    return
                
                # Sort points for a smoother demonstration
                sorted_points = sorted(boundary_points, key=lambda p: (p[0], p[1]))
                
                print(f"\nDemonstrating {len(sorted_points)} reachable points for {robot_type} robot...")
                print("Press ESC to stop demonstration")
                
                # Move to an initial neutral position
                if robot_type == "flat":
                    neutral_pos = np.array([0.0, -1.57, 0.0, -1.57, 0.0, 0.0])
                    for i, joint_id in enumerate(analyzer.flat_ik_solver.joint_ids):
                        data.qpos[joint_id] = neutral_pos[i]
                else:
                    neutral_pos = np.array([np.pi/2, -1.57, 0.0, -1.57, 0.0, 0.0])
                    for i, joint_id in enumerate(analyzer.perp_ik_solver.joint_ids):
                        data.qpos[joint_id] = neutral_pos[i]
                        
                mujoco.mj_forward(model, data)
                
                # Render the initial position
                if not visualized_pause(0.5):
                    return
                
                # Move to each boundary point
                success_count = 0
                for i, (y, z) in enumerate(sorted_points):
                    # Create target position
                    target_pos = np.array([analyzer.wall_x, y, z])
                    
                    # Attempt to move robot to point with perpendicular tool
                    print(f"\nAttempting point {i+1}/{len(sorted_points)}: {target_pos}")
                    
                    # Select the appropriate IK solver
                    ik_solver = analyzer.flat_ik_solver if robot_type == "flat" else analyzer.perp_ik_solver
                    
                    # Try to solve IK
                    q, success = ik_solver.solve_ik(target_pos)
                    
                    if success:
                        # Set the joint positions
                        for j, joint_id in enumerate(ik_solver.joint_ids):
                            data.qpos[joint_id] = q[j]
                        
                        # Update forward kinematics
                        mujoco.mj_forward(model, data)
                        
                        # Move the appropriate reach point marker for visualization
                        if robot_type == "flat":
                            data.site_xpos[analyzer.flat_reach_point_id] = target_pos
                        else:
                            data.site_xpos[analyzer.perp_reach_point_id] = target_pos
                        
                        success_count += 1
                        print(f"Point {i+1}/{len(sorted_points)}: Success")
                        
                        # Render and pause to visualize
                        if not visualized_pause(0.5):
                            break
                    else:
                        print(f"Point {i+1}/{len(sorted_points)}: Failed")
                
                print(f"\nDemonstration completed: {success_count}/{len(sorted_points)} points reached successfully")
                
                # Reset robot to a neutral position
                if robot_type == "flat":
                    neutral_pos = np.array([0.0, -1.57, 0.0, -1.57, 0.0, 0.0])
                    for i, joint_id in enumerate(analyzer.flat_ik_solver.joint_ids):
                        data.qpos[joint_id] = neutral_pos[i]
                else:
                    neutral_pos = np.array([np.pi/2, -1.57, 0.0, -1.57, 0.0, 0.0])
                    for i, joint_id in enumerate(analyzer.perp_ik_solver.joint_ids):
                        data.qpos[joint_id] = neutral_pos[i]
                
                mujoco.mj_forward(model, data)
                visualized_pause(0.5)
            
            # Wait for user input before starting demonstration
            print("\nPress Enter to start the demonstration, or 'q' to skip: ", end='', flush=True)
            user_input = input().strip().lower()
            if user_input != 'q':
                # Run the demonstrations
                if robot_type == "both":
                    print("\nDemonstrating flat robot boundary...")
                    demo_boundary_with_viz("flat")
                    print("\nPress Enter to continue with perpendicular robot, or 'q' to skip: ", end='', flush=True)
                    user_input = input().strip().lower()
                    if user_input != 'q':
                        print("\nDemonstrating perpendicular robot boundary...")
                        demo_boundary_with_viz("perp")
                else:
                    print(f"\nDemonstrating {robot_type} robot boundary...")
                    demo_boundary_with_viz(robot_type)
        
        # Main interactive loop
        print("\nInteractive mode: Use mouse to navigate, ESC to exit")
        while interactive and window:
            if not render_frame(window, model, data, scene, camera, option, context):
                break
            
            # Refresh plots if they exist to keep them responsive
            if visualize:
                plt.pause(0.01)
        
        # Clean up MuJoCo visualization
        from mujoco.glfw import glfw
        if window:
            glfw.terminate()
        print("Viewer closed")
        
        # Keep plots open if they were created
        if visualize:
            print("Plots remain open. Close plot windows to exit completely.")
            plt.ioff()  # Turn off interactive mode
            plt.show()  # This will block until all plot windows are closed
    
    else:
        # Non-interactive mode
        print("\nRunning in non-interactive (batch) mode...")
        analyzer.analyze_reachability(progress_callback=progress_callback)
        
        # Generate and save plots
        print("\nGenerating plots...")
        os.makedirs("results", exist_ok=True)
        
        # Plot individual reachability maps
        reachability_fig = analyzer.plot_reachability_maps()
        reachability_fig.savefig("results/reachability_maps.png", dpi=300, bbox_inches='tight')
        
        # Plot comparison
        comparison_fig = analyzer.plot_comparison()
        comparison_fig.savefig("results/reachability_comparison.png", dpi=300, bbox_inches='tight')
        
        print("Analysis complete. Results saved to 'results' directory.")


def main():
    parser = argparse.ArgumentParser(description="UR10 Reachability Analysis Tool")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, 
                       help=f"Path to MuJoCo model XML file (default: {DEFAULT_MODEL})")
    parser.add_argument("--non-interactive", action="store_true", 
                       help="Run in non-interactive (batch) mode")
    parser.add_argument("--no-visualization", action="store_true", 
                       help="Disable result visualization")
    parser.add_argument("--demo-boundary", action="store_true", 
                       help="Demonstrate the reachable boundary by moving the robot")
    parser.add_argument("--save-plots", action="store_true", 
                       help="Save the plots to the 'results' directory")
    parser.add_argument("--y-steps", type=int, default=15, 
                       help="Number of sampling steps along Y axis (default: 15)")
    parser.add_argument("--z-steps", type=int, default=15, 
                       help="Number of sampling steps along Z axis (default: 15)")
    parser.add_argument("--robot", type=str, choices=["flat", "perp", "both"], default="both", 
                       help="Which robot to analyze (default: both)")
    
    args = parser.parse_args()
    
    # Run the analysis
    run_analysis(
        model_path=args.model,
        interactive=not args.non_interactive,
        visualize=not args.no_visualization,
        demo_boundary=args.demo_boundary,
        save_plots=args.save_plots,
        y_steps=args.y_steps,
        z_steps=args.z_steps,
        robot_type=args.robot
    )

if __name__ == "__main__":
    main() 