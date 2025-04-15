#!/usr/bin/env python3
import os
import time
import numpy as np
import mujoco
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull
import argparse

# Set random seed for reproducibility
np.random.seed(42)

# Wall parameters
WALL_X = 1.0  # Wall X position
WALL_TOLERANCE = 0.05  # Distance tolerance to consider a point at the wall
PERP_TOLERANCE = 0.2  # Tolerance for perpendicularity to wall (cosine similarity)

def create_marker(model, data, pos, size, rgba, name):
    """Create a visual marker in the scene."""
    marker_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if marker_site_id >= 0:
        data.site_pos[marker_site_id] = pos
        data.site_size[marker_site_id] = np.array([size, size, size])
        data.site_rgba[marker_site_id] = rgba

def is_perpendicular_to_wall(rot_matrix, tolerance=PERP_TOLERANCE):
    """Check if the end effector is perpendicular to the wall."""
    # Z axis of the end effector
    z_axis = rot_matrix[:, 2]
    
    # Wall normal (pointing in positive X)
    wall_normal = np.array([1, 0, 0])
    
    # Cosine similarity (dot product of unit vectors)
    cos_sim = np.abs(np.dot(z_axis, wall_normal))
    
    # We want vectors to be close to parallel (cos_sim close to 1)
    return cos_sim > (1 - tolerance)

def is_at_wall(pos, tolerance=WALL_TOLERANCE):
    """Check if a position is at the wall."""
    return abs(pos[0] - WALL_X) < tolerance

def get_end_effector_pose(model, data, body_name):
    """Get end effector position and orientation matrix."""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    
    if body_id >= 0:
        # Get position - use mujoco.mj_data functions
        pos = np.zeros(3)
        rot_mat = np.zeros((3, 3))
        
        # Get body position and orientation using lower level functions
        mujoco.mj_objectPosition(model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, pos)
        mujoco.mj_objectRotation(model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, rot_mat.flatten())
        
        # Reshape to proper 3x3 matrix
        rot_mat = rot_mat.reshape(3, 3)
        
        return pos, rot_mat
    else:
        return None, None

def sample_joint_positions(num_samples=1000):
    """Generate random joint configurations."""
    # Joint limits for UR10
    joint_limits = np.array([
        [-np.pi, np.pi],       # Shoulder pan
        [-np.pi, 0],           # Shoulder lift
        [0, np.pi],            # Elbow
        [-np.pi, np.pi],       # Wrist 1
        [-np.pi, np.pi],       # Wrist 2
        [-np.pi, np.pi]        # Wrist 3
    ])
    
    # Sample uniformly from joint limits
    samples = np.random.uniform(
        joint_limits[:, 0],
        joint_limits[:, 1],
        size=(num_samples, 6)
    )
    
    return samples

def set_robot_joints(model, data, config_name, joint_positions):
    """Set joint positions for a specific robot configuration."""
    for i, joint_name in enumerate([
        f"{config_name}_shoulder_pan_joint",
        f"{config_name}_shoulder_lift_joint",
        f"{config_name}_elbow_joint",
        f"{config_name}_wrist_1_joint",
        f"{config_name}_wrist_2_joint",
        f"{config_name}_wrist_3_joint"
    ]):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id >= 0:
            data.qpos[joint_id] = joint_positions[i]
    
    # Update simulation
    mujoco.mj_forward(model, data)

def analyze_reachability(model, data, config_name, num_samples=5000):
    """Analyze reachable points on the wall for a robot configuration."""
    print(f"Analyzing reachability for {config_name} configuration...")
    
    # Sample joint configurations
    joint_samples = sample_joint_positions(num_samples)
    
    # Store reachable points on the wall
    reachable_points = []
    
    # End effector body name
    ee_body_name = f"{config_name}_wrist_3_link"
    
    for i, joints in enumerate(joint_samples):
        if i % 500 == 0:
            print(f"  Processing sample {i}/{num_samples}")
        
        # Set joints
        set_robot_joints(model, data, config_name, joints)
        
        # Get end effector pose
        ee_pos, ee_rot = get_end_effector_pose(model, data, ee_body_name)
        
        # Check if at wall and perpendicular
        if ee_pos is not None and is_at_wall(ee_pos) and is_perpendicular_to_wall(ee_rot):
            reachable_points.append(ee_pos)
    
    reachable_points = np.array(reachable_points) if reachable_points else np.zeros((0, 3))
    print(f"Found {len(reachable_points)} reachable points for {config_name} configuration")
    
    return reachable_points

def create_wall_heatmap(flat_points, perp_points, resolution=100):
    """Create a 2D heatmap visualization of the wall coverage."""
    # Define wall boundaries
    y_min, y_max = -2.0, 2.0
    z_min, z_max = 0.0, 3.0
    
    # Create 2D histogram for each configuration
    y_bins = np.linspace(y_min, y_max, resolution)
    z_bins = np.linspace(z_min, z_max, resolution)
    
    flat_hist, _, _ = np.histogram2d(
        flat_points[:, 1] if len(flat_points) > 0 else np.array([]),
        flat_points[:, 2] if len(flat_points) > 0 else np.array([]),
        bins=[y_bins, z_bins]
    )
    
    perp_hist, _, _ = np.histogram2d(
        perp_points[:, 1] if len(perp_points) > 0 else np.array([]),
        perp_points[:, 2] if len(perp_points) > 0 else np.array([]),
        bins=[y_bins, z_bins]
    )
    
    # Normalize histograms
    flat_hist = (flat_hist > 0).astype(float)
    perp_hist = (perp_hist > 0).astype(float)
    
    # Create a combined visualization
    # Red channel = flat configuration
    # Blue channel = perpendicular configuration
    # Purple = both configurations
    rgb_hist = np.zeros((resolution, resolution, 3))
    rgb_hist[:, :, 0] = flat_hist.T  # Red channel
    rgb_hist[:, :, 2] = perp_hist.T  # Blue channel
    
    # Calculate coverage areas
    flat_coverage = np.sum(flat_hist) / (resolution * resolution)
    perp_coverage = np.sum(perp_hist) / (resolution * resolution)
    overlap = np.sum((flat_hist > 0) & (perp_hist > 0)) / (resolution * resolution)
    
    # Create the heatmap figure
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(rgb_hist, origin='lower', extent=[y_min, y_max, z_min, z_max], interpolation='nearest')
    
    # Add labels and title
    ax.set_xlabel('Y (m)')
    ax.set_ylabel('Z (m)')
    ax.set_title('Wall Coverage Comparison')
    
    # Add a custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', edgecolor='r', label=f'Flat Mount ({flat_coverage:.1%} coverage)'),
        Patch(facecolor='blue', edgecolor='b', label=f'Perpendicular Mount ({perp_coverage:.1%} coverage)'),
        Patch(facecolor='purple', edgecolor='k', label=f'Both ({overlap:.1%} overlap)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Add grid lines
    ax.grid(True, color='white', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # Determine which configuration has better coverage
    if flat_coverage > perp_coverage:
        conclusion = "Flat mounting provides better wall coverage"
    elif perp_coverage > flat_coverage:
        conclusion = "Perpendicular mounting provides better wall coverage"
    else:
        conclusion = "Both configurations provide similar wall coverage"
    
    # Add conclusion text
    ax.text(y_min + 0.1, z_min + 0.1, conclusion, color='white', 
            bbox=dict(facecolor='black', alpha=0.7))
    
    return fig, flat_coverage, perp_coverage

def visualize_reachable_areas(model_path, output_dir, num_samples=5000):
    """Visualize reachable areas for both configurations."""
    # Load MuJoCo model
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Analyze flat configuration
    flat_points = analyze_reachability(model, data, "flat", num_samples)
    np.save(os.path.join(output_dir, "flat_reachable_points.npy"), flat_points)
    
    # Analyze perpendicular configuration
    perp_points = analyze_reachability(model, data, "perp", num_samples)
    np.save(os.path.join(output_dir, "perp_reachable_points.npy"), perp_points)
    
    # Create and save heatmap visualization
    heatmap_fig, flat_coverage, perp_coverage = create_wall_heatmap(flat_points, perp_points)
    heatmap_fig.savefig(os.path.join(output_dir, "wall_coverage_heatmap.png"), dpi=300, bbox_inches='tight')
    
    # Create a summary report
    with open(os.path.join(output_dir, "coverage_report.txt"), "w") as f:
        f.write("# UR10 Wall Coverage Analysis\n\n")
        f.write(f"Samples per configuration: {num_samples}\n\n")
        
        f.write("## Flat Mounting Configuration\n")
        f.write(f"Reachable points: {len(flat_points)}\n")
        f.write(f"Wall coverage: {flat_coverage:.2%}\n\n")
        
        f.write("## Perpendicular Mounting Configuration\n")
        f.write(f"Reachable points: {len(perp_points)}\n")
        f.write(f"Wall coverage: {perp_coverage:.2%}\n\n")
        
        f.write("## Conclusion\n")
        if flat_coverage > perp_coverage:
            f.write("Flat mounting provides better wall coverage\n")
            f.write(f"Improvement: {(flat_coverage/perp_coverage - 1):.2%}\n")
        elif perp_coverage > flat_coverage:
            f.write("Perpendicular mounting provides better wall coverage\n")
            f.write(f"Improvement: {(perp_coverage/flat_coverage - 1):.2%}\n")
        else:
            f.write("Both configurations provide similar wall coverage\n")
    
    print(f"Analysis complete. Results saved to {output_dir}")
    
    return flat_points, perp_points

def interactive_visualization(model_path, flat_points, perp_points):
    """Interactive visualization with MuJoCo."""
    # Load model
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    
    # Initialize GLFW
    mujoco.mj_resetData(model, data)
    window = mujoco.glfw.init_glfw("UR10 Reach Analysis", model)
    mujoco.mjv_defaultCamera(mujoco.MjvCamera())
    
    # Create scene and context
    scene = mujoco.MjvScene(model, maxgeom=10000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
    
    # Create camera
    camera = mujoco.MjvCamera()
    camera.lookat[0] = 0.0
    camera.lookat[1] = 0.0
    camera.lookat[2] = 1.0
    camera.distance = 4.0
    camera.azimuth = 90
    camera.elevation = -20
    
    # Option settings
    option = mujoco.MjvOption()
    
    # Add reachable points as small spheres on the wall
    def add_points_to_scene(points, color):
        for point in points[:500]:  # Limit to 500 points to avoid performance issues
            pos = point.copy()
            
            # Add a small sphere at the position
            sphere = mujoco.MjvGeom()
            sphere.type = mujoco.mjtGeom.mjGEOM_SPHERE
            sphere.size[:] = [0.01, 0, 0]  # Small sphere
            sphere.pos[:] = pos
            sphere.rgba[:] = color
            sphere.emission = 0.5
            scene.add_geom(sphere)
    
    # Demo mode settings
    demo_mode = True
    demo_counter = 0
    demo_joints = []
    
    # Generate some demonstration joint configurations
    if len(flat_points) > 0:
        # For flat robot demo, find a few good configurations
        flat_demo_indices = np.random.choice(len(flat_points), min(10, len(flat_points)), replace=False)
        flat_demo_points = flat_points[flat_demo_indices]
        
        # We need to reverse engineer joint configurations for these points
        # For simplicity, we'll just sample and keep those that reach close to the demo points
        flat_joints = []
        samples = sample_joint_positions(2000)
        
        for joints in samples:
            set_robot_joints(model, data, "flat", joints)
            ee_pos, ee_rot = get_end_effector_pose(model, data, "flat_wrist_3_link")
            
            if ee_pos is not None and is_at_wall(ee_pos) and is_perpendicular_to_wall(ee_rot):
                flat_joints.append(joints)
                if len(flat_joints) >= 10:
                    break
        
        demo_joints.append(("flat", flat_joints))
    
    if len(perp_points) > 0:
        # Same for perpendicular robot
        perp_demo_indices = np.random.choice(len(perp_points), min(10, len(perp_points)), replace=False)
        perp_demo_points = perp_points[perp_demo_indices]
        
        perp_joints = []
        samples = sample_joint_positions(2000)
        
        for joints in samples:
            set_robot_joints(model, data, "perp", joints)
            ee_pos, ee_rot = get_end_effector_pose(model, data, "perp_wrist_3_link")
            
            if ee_pos is not None and is_at_wall(ee_pos) and is_perpendicular_to_wall(ee_rot):
                perp_joints.append(joints)
                if len(perp_joints) >= 10:
                    break
        
        demo_joints.append(("perp", perp_joints))
    
    # Main visualization loop
    print("\nInteractive Visualization Controls:")
    print("  ESC: Quit")
    print("  Space: Toggle demo mode")
    print("  R: Reset view")
    print("  A: Toggle reachable point visualization")
    
    show_points = True
    current_demo_config = 0
    last_update_time = time.time()
    
    while not mujoco.glfw.window_should_close(window):
        time_now = time.time()
        
        # Demo mode: cycle through different robot positions
        if demo_mode and time_now - last_update_time > 2.0:  # Change every 2 seconds
            if demo_joints:
                config_name, joints_list = demo_joints[current_demo_config]
                
                if joints_list:
                    joint_pos = joints_list[demo_counter % len(joints_list)]
                    set_robot_joints(model, data, config_name, joint_pos)
                    
                    demo_counter += 1
                    if demo_counter % len(joints_list) == 0:
                        current_demo_config = (current_demo_config + 1) % len(demo_joints)
                
            last_update_time = time_now
        
        # Update scene
        mujoco.mjv_updateScene(
            model, data, option, None, camera,
            mujoco.mjtCatBit.mjCAT_ALL.value, scene
        )
        
        # Add reachable points if enabled
        if show_points:
            add_points_to_scene(flat_points, [1, 0, 0, 0.5])  # Red for flat
            add_points_to_scene(perp_points, [0, 0, 1, 0.5])  # Blue for perpendicular
        
        # Render scene
        mujoco.mjr_render(scene, 0, 0, context.width, context.height, context)
        
        # Process events
        mujoco.glfw.poll_events()
        
        # Check for keyboard input
        if mujoco.glfw.get_key(window, mujoco.glfw.KEY_ESCAPE) == mujoco.glfw.PRESS:
            break
        
        if mujoco.glfw.get_key(window, mujoco.glfw.KEY_SPACE) == mujoco.glfw.PRESS:
            # Toggle demo mode
            demo_mode = not demo_mode
            time.sleep(0.3)  # Simple debounce
        
        if mujoco.glfw.get_key(window, mujoco.glfw.KEY_R) == mujoco.glfw.PRESS:
            # Reset view
            camera.lookat[0] = 0.0
            camera.lookat[1] = 0.0
            camera.lookat[2] = 1.0
            camera.distance = 4.0
            camera.azimuth = 90
            camera.elevation = -20
            time.sleep(0.3)  # Simple debounce
        
        if mujoco.glfw.get_key(window, mujoco.glfw.KEY_A) == mujoco.glfw.PRESS:
            # Toggle points visualization
            show_points = not show_points
            time.sleep(0.3)  # Simple debounce
        
        # Swap buffers
        mujoco.glfw.swap_buffers(window)
    
    # Clean up
    mujoco.glfw.terminate()

def main():
    parser = argparse.ArgumentParser(description="UR10 Reach Analysis Visualization")
    parser.add_argument("--model", type=str, default="src/models/reach_comparison.xml", 
                       help="Path to MuJoCo model XML file")
    parser.add_argument("--output", type=str, default="results", 
                       help="Output directory for results")
    parser.add_argument("--samples", type=int, default=5000, 
                       help="Number of joint configurations to sample")
    parser.add_argument("--interactive", action="store_true", 
                       help="Run interactive visualization")
    parser.add_argument("--load", action="store_true", 
                       help="Load previously computed results instead of recomputing")
    
    args = parser.parse_args()
    
    model_path = Path(args.model)
    output_dir = Path(args.output)
    
    # Load or compute reachable points
    if args.load and os.path.exists(os.path.join(output_dir, "flat_reachable_points.npy")):
        print("Loading previously computed results...")
        flat_points = np.load(os.path.join(output_dir, "flat_reachable_points.npy"))
        perp_points = np.load(os.path.join(output_dir, "perp_reachable_points.npy"))
        
        print(f"Loaded {len(flat_points)} flat mount points and {len(perp_points)} perpendicular mount points")
        
        # Recreate heatmap
        heatmap_fig, flat_coverage, perp_coverage = create_wall_heatmap(flat_points, perp_points)
        heatmap_fig.savefig(os.path.join(output_dir, "wall_coverage_heatmap.png"), dpi=300, bbox_inches='tight')
    else:
        # Compute reachable points
        flat_points, perp_points = visualize_reachable_areas(model_path, output_dir, args.samples)
    
    # Run interactive visualization if requested
    if args.interactive:
        print("Starting interactive visualization...")
        interactive_visualization(model_path, flat_points, perp_points)

if __name__ == "__main__":
    main() 