#!/usr/bin/env python3
import numpy as np
import time
import mujoco
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.simulation.ik_solver import UR10eIKSolver

class ReachabilityAnalyzer:
    """Class to analyze and visualize the reachable workspace of UR10e robots."""
    
    def __init__(self, model, data):
        """
        Initialize the reachability analyzer.
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
        """
        self.model = model
        self.data = data
        
        # Create IK solvers for both robots
        self.flat_ik_solver = UR10eIKSolver(model, data, robot_prefix="flat_")
        self.perp_ik_solver = UR10eIKSolver(model, data, robot_prefix="perp_")
        
        # Get positions of robot bases
        flat_base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "flat_robot_base")
        perp_base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "perp_robot_base")
        
        # Reset simulation to get body positions
        mujoco.mj_forward(self.model, self.data)
        
        # Get body positions
        self.flat_base_pos = self.data.xpos[flat_base_id].copy()
        self.perp_base_pos = self.data.xpos[perp_base_id].copy()
        
        # Wall position
        self.wall_x = 1.2  # X coordinate of the wall
        
        # Reachability maps
        self.flat_reach_map = None
        self.perp_reach_map = None
        
        # Grid parameters
        self.y_grid = None
        self.z_grid = None
        
        # Reach point site IDs for visualization
        self.flat_reach_point_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "flat_reach_point")
        self.perp_reach_point_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "perp_reach_point")
        
        print(f"Robot base positions:")
        print(f"  Flat robot: {self.flat_base_pos}")
        print(f"  Perp robot: {self.perp_base_pos}")
    
    def setup_grid(self, y_min=-2.0, y_max=2.0, z_min=0.1, z_max=3.0, y_steps=30, z_steps=30):
        """
        Set up a grid of points on the wall to test for reachability.
        
        Args:
            y_min: Minimum Y coordinate
            y_max: Maximum Y coordinate
            z_min: Minimum Z coordinate
            z_max: Maximum Z coordinate
            y_steps: Number of steps along Y axis
            z_steps: Number of steps along Z axis
        """
        self.y_grid = np.linspace(y_min, y_max, y_steps)
        self.z_grid = np.linspace(z_min, z_max, z_steps)
        
        # Initialize reachability maps
        self.flat_reach_map = np.zeros((y_steps, z_steps), dtype=bool)
        self.perp_reach_map = np.zeros((y_steps, z_steps), dtype=bool)
        
        print(f"Grid setup: {y_steps} x {z_steps} points")
        print(f"Y range: {y_min:.2f} to {y_max:.2f}")
        print(f"Z range: {z_min:.2f} to {z_max:.2f}")
    
    def analyze_reachability(self, progress_callback=None, robot_type="both"):
        """
        Analyze the reachability of both robots across the grid.
        
        Args:
            progress_callback: Optional callback function to report progress
                              Called with (y_index, z_index, total_y, total_z)
            robot_type: Which robot to analyze ("flat", "perp", or "both")
        
        Returns:
            flat_reach_map: 2D boolean array of reachability for flat robot
            perp_reach_map: 2D boolean array of reachability for perp robot
        """
        if self.y_grid is None or self.z_grid is None:
            raise ValueError("Grid not set up. Call setup_grid() first.")
        
        total_points = len(self.y_grid) * len(self.z_grid)
        print(f"Analyzing reachability for {total_points} points...")
        start_time = time.time()
        
        # Process the grid
        for y_idx, y in enumerate(self.y_grid):
            for z_idx, z in enumerate(self.z_grid):
                # Create target point on wall
                target_pos = np.array([self.wall_x, y, z])
                
                # Check reachability based on specified robot type
                if robot_type in ["flat", "both"]:
                    # Check if within a more generous reach distance for flat robot
                    flat_distance = np.linalg.norm(target_pos - self.flat_base_pos)
                    if flat_distance < 2.0:  # UR10e has ~1.3m reach, be generous
                        # Check reachability for flat robot
                        flat_reachable, _ = self.flat_ik_solver.is_point_reachable(target_pos)
                        self.flat_reach_map[y_idx, z_idx] = flat_reachable
                
                if robot_type in ["perp", "both"]:
                    # Check if within a more generous reach distance for perp robot
                    perp_distance = np.linalg.norm(target_pos - self.perp_base_pos)
                    if perp_distance < 2.0:  # UR10e has ~1.3m reach, be generous
                        # Check reachability for perp robot
                        perp_reachable, _ = self.perp_ik_solver.is_point_reachable(target_pos)
                        self.perp_reach_map[y_idx, z_idx] = perp_reachable
                
                # Report progress if callback is provided
                if progress_callback:
                    progress_callback(y_idx, z_idx, len(self.y_grid), len(self.z_grid))
        
        elapsed_time = time.time() - start_time
        print(f"Reachability analysis completed in {elapsed_time:.2f} seconds")
        
        # Calculate statistics
        flat_reachable_points = np.sum(self.flat_reach_map)
        perp_reachable_points = np.sum(self.perp_reach_map)
        
        print(f"Flat robot: {flat_reachable_points} reachable points ({flat_reachable_points / total_points * 100:.1f}%)")
        print(f"Perp robot: {perp_reachable_points} reachable points ({perp_reachable_points / total_points * 100:.1f}%)")
        
        return self.flat_reach_map, self.perp_reach_map
    
    def plot_reachability_maps(self, save_path=None):
        """
        Plot the reachability maps for both robots.
        
        Args:
            save_path: Optional path to save the figure
        
        Returns:
            fig: Matplotlib figure
        """
        if self.flat_reach_map is None or self.perp_reach_map is None:
            raise ValueError("Reachability analysis not performed. Call analyze_reachability() first.")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Create custom colormaps
        flat_cmap = LinearSegmentedColormap.from_list('flat_cmap', [(1, 1, 1, 0), (1, 0, 0, 1)])
        perp_cmap = LinearSegmentedColormap.from_list('perp_cmap', [(1, 1, 1, 0), (0, 0, 1, 1)])
        
        # Plot flat robot reachability
        y_mesh, z_mesh = np.meshgrid(self.y_grid, self.z_grid, indexing='ij')
        flat_reach = ax1.pcolormesh(y_mesh, z_mesh, self.flat_reach_map, cmap=flat_cmap, shading='auto')
        ax1.set_title("Flat Mounted Robot Reachability")
        ax1.set_xlabel("Y Position (m)")
        ax1.set_ylabel("Z Position (m)")
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Add flat robot position marker
        ax1.plot(self.flat_base_pos[1], self.flat_base_pos[2], 'kx', markersize=10, label='Robot Base')
        
        # Plot perp robot reachability
        perp_reach = ax2.pcolormesh(y_mesh, z_mesh, self.perp_reach_map, cmap=perp_cmap, shading='auto')
        ax2.set_title("Perpendicular Mounted Robot Reachability")
        ax2.set_xlabel("Y Position (m)")
        ax2.set_ylabel("Z Position (m)")
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Add perp robot position marker
        ax2.plot(self.perp_base_pos[1], self.perp_base_pos[2], 'kx', markersize=10, label='Robot Base')
        
        # Add colorbar
        fig.colorbar(flat_reach, ax=ax1, label="Reachable")
        fig.colorbar(perp_reach, ax=ax2, label="Reachable")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        return fig
    
    def plot_comparison(self, save_path=None):
        """
        Plot a comparison of both reachability maps on the same axes.
        
        Args:
            save_path: Optional path to save the figure
        
        Returns:
            fig: Matplotlib figure
        """
        if self.flat_reach_map is None or self.perp_reach_map is None:
            raise ValueError("Reachability analysis not performed. Call analyze_reachability() first.")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create a combined visualization
        # 1 = flat only, 2 = perp only, 3 = both
        combined_map = self.flat_reach_map.astype(int) + 2 * self.perp_reach_map.astype(int)
        
        # Create custom colormap for combined visualization
        colors = [(1, 1, 1, 0),      # 0: neither (transparent)
                 (1, 0, 0, 1),       # 1: flat only (red)
                 (0, 0, 1, 1),       # 2: perp only (blue)
                 (0.7, 0, 0.7, 1)]   # 3: both (purple)
        combined_cmap = LinearSegmentedColormap.from_list('combined_cmap', colors, N=4)
        
        # Plot combined reachability
        y_mesh, z_mesh = np.meshgrid(self.y_grid, self.z_grid, indexing='ij')
        combined_reach = ax.pcolormesh(y_mesh, z_mesh, combined_map, cmap=combined_cmap, shading='auto', vmin=0, vmax=3)
        
        ax.set_title("Reachability Comparison")
        ax.set_xlabel("Y Position (m)")
        ax.set_ylabel("Z Position (m)")
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add robot base positions
        ax.plot(self.flat_base_pos[1], self.flat_base_pos[2], 'kx', markersize=10, label='Flat Robot Base')
        ax.plot(self.perp_base_pos[1], self.perp_base_pos[2], 'k+', markersize=10, label='Perp Robot Base')
        ax.legend()
        
        # Add wall line
        wall_y = np.array([self.y_grid.min(), self.y_grid.max()])
        wall_z = np.array([self.z_grid.min(), self.z_grid.max()])
        ax.plot([wall_y[0], wall_y[0], wall_y[1], wall_y[1], wall_y[0]],
                [wall_z[0], wall_z[1], wall_z[1], wall_z[0], wall_z[0]],
                'k--', alpha=0.5, label='Wall Boundary')
        
        # Add custom legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='Flat Robot Only'),
            Patch(facecolor='blue', label='Perp Robot Only'),
            Patch(facecolor='purple', label='Both Robots')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Calculate statistics
        flat_only = np.sum((combined_map == 1).astype(int))
        perp_only = np.sum((combined_map == 2).astype(int))
        both = np.sum((combined_map == 3).astype(int))
        total = flat_only + perp_only + both
        
        # Add statistics text
        if total > 0:
            stats_text = f"Flat Only: {flat_only} ({flat_only/total*100:.1f}%)\n"
            stats_text += f"Perp Only: {perp_only} ({perp_only/total*100:.1f}%)\n"
            stats_text += f"Both: {both} ({both/total*100:.1f}%)"
            ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, 
                    bbox=dict(facecolor='white', alpha=0.7))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        return fig
    
    def find_reachable_boundary(self, robot_type="flat"):
        """
        Find the boundary points of the reachable workspace.
        
        Args:
            robot_type: Which robot to analyze ("flat" or "perp")
        
        Returns:
            boundary_points: List of (y, z) boundary points
        """
        if self.flat_reach_map is None or self.perp_reach_map is None:
            raise ValueError("Reachability analysis not performed. Call analyze_reachability() first.")
        
        # Select the appropriate reachability map
        reach_map = self.flat_reach_map if robot_type == "flat" else self.perp_reach_map
        
        # Find reachable points
        reachable_indices = np.argwhere(reach_map)
        
        # Convert indices to world coordinates
        boundary_points = []
        for idx in reachable_indices:
            y = self.y_grid[idx[0]]
            z = self.z_grid[idx[1]]
            boundary_points.append((y, z))
        
        return boundary_points
    
    def move_robot_to_point(self, target_pos, robot_type="flat"):
        """
        Move the robot to a target point with the tool perpendicular to the wall.
        
        Args:
            target_pos: Target position [x, y, z]
            robot_type: Which robot to move ("flat" or "perp")
        
        Returns:
            success: True if the robot successfully moved to the point
        """
        # Select the appropriate IK solver
        ik_solver = self.flat_ik_solver if robot_type == "flat" else self.perp_ik_solver
        
        # Try to solve IK with strong perpendicularity constraint
        q, success = ik_solver.solve_ik(target_pos)
        
        if success:
            # Set the joint positions
            for i, joint_id in enumerate(ik_solver.joint_ids):
                self.data.qpos[joint_id] = q[i]
            
            # Update forward kinematics
            mujoco.mj_forward(self.model, self.data)
            
            # Move the appropriate reach point marker for visualization
            if robot_type == "flat":
                self.data.site_xpos[self.flat_reach_point_id] = target_pos
            else:
                self.data.site_xpos[self.perp_reach_point_id] = target_pos
                
            # Verify that the desired perpendicularity is achieved
            curr_pos, curr_rot = ik_solver.get_tool_pose()
            is_perp = ik_solver.is_perpendicular_to_wall(curr_rot)
            
            if not is_perp:
                print(f"Warning: Tool is not perpendicular to wall at {target_pos}")
                # Try to fix by applying additional wrist adjustments
                self.fix_perpendicularity(robot_type)
                return False  # Report failure if perpendicularity not achieved
            
            # Log success with current tool orientation
            tool_rot = self.data.xmat[ik_solver.tool_id].reshape(3, 3).copy()
            wall_normal = np.array([-1.0, 0.0, 0.0])
            alignment = np.abs(np.dot(tool_rot[:, 1], wall_normal))
            print(f"Success at {target_pos}: Alignment = {alignment:.4f}")
            
            return True
        else:
            print(f"Failed to find IK solution for {target_pos}")
            return False
    
    def fix_perpendicularity(self, robot_type="flat"):
        """
        Attempt to fix tool perpendicularity by adjusting wrist joints.
        
        Args:
            robot_type: Which robot to adjust ("flat" or "perp")
        """
        # Select the appropriate IK solver
        ik_solver = self.flat_ik_solver if robot_type == "flat" else self.perp_ik_solver
        
        # Get current joint positions
        current_q = np.array([self.data.qpos[joint_id] for joint_id in ik_solver.joint_ids])
        
        # Try different wrist orientations
        best_alignment = -1.0
        best_q = None
        
        for wrist1 in [-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2]:
            for wrist2 in [-np.pi/4, 0, np.pi/4]:
                for wrist3 in [-np.pi/4, 0, np.pi/4]:
                    test_q = current_q.copy()
                    test_q[3] = wrist1  # wrist1
                    test_q[4] = wrist2  # wrist2
                    test_q[5] = wrist3  # wrist3
                    
                    # Test this configuration
                    for i, joint_id in enumerate(ik_solver.joint_ids):
                        self.data.qpos[joint_id] = test_q[i]
                    
                    mujoco.mj_forward(self.model, self.data)
                    
                    # Check perpendicularity
                    tool_rot = self.data.xmat[ik_solver.tool_id].reshape(3, 3).copy()
                    wall_normal = np.array([-1.0, 0.0, 0.0])
                    alignment = np.abs(np.dot(tool_rot[:, 1], wall_normal))
                    
                    if alignment > best_alignment:
                        best_alignment = alignment
                        best_q = test_q.copy()
        
        # Apply best configuration if found
        if best_q is not None and best_alignment > 0.9:  # Only if we found good alignment
            for i, joint_id in enumerate(ik_solver.joint_ids):
                self.data.qpos[joint_id] = best_q[i]
            
            mujoco.mj_forward(self.model, self.data)
            print(f"Fixed perpendicularity. New alignment: {best_alignment:.4f}")
        else:
            print("Could not find good perpendicular orientation")
    
    def demonstrate_boundary(self, robot_type="flat", pause_time=0.5):
        """
        Demonstrate the reachable boundary by moving the robot to boundary points.
        NOTE: This method is deprecated as real-time visualization is now handled in run_reach_analysis.py
        
        Args:
            robot_type: Which robot to demonstrate ("flat" or "perp")
            pause_time: Time to pause at each point (seconds)
        """
        print("NOTE: This method is deprecated. Real-time visualization now handled in run_reach_analysis.py")
        
        # Get boundary points
        boundary_points = self.find_reachable_boundary(robot_type)
        
        if not boundary_points:
            print(f"No reachable points found for {robot_type} robot.")
            return
        
        # Sort points for a smoother demonstration
        # Sort by Y then Z for a more natural movement pattern
        sorted_points = sorted(boundary_points, key=lambda p: (p[0], p[1]))
        
        print(f"Demonstrating {len(sorted_points)} reachable points for {robot_type} robot...")
        print("Press ESC to stop demonstration")
        
        # Move to an initial neutral position
        if robot_type == "flat":
            neutral_pos = np.array([0.0, -1.57, 0.0, -1.57, 0.0, 0.0])
            for i, joint_id in enumerate(self.flat_ik_solver.joint_ids):
                self.data.qpos[joint_id] = neutral_pos[i]
        else:
            neutral_pos = np.array([np.pi/2, -1.57, 0.0, -1.57, 0.0, 0.0])
            for i, joint_id in enumerate(self.perp_ik_solver.joint_ids):
                self.data.qpos[joint_id] = neutral_pos[i]
                
        mujoco.mj_forward(self.model, self.data)
        time.sleep(pause_time)
        
        # Move to each boundary point
        success_count = 0
        for i, (y, z) in enumerate(sorted_points):
            # Create target position
            target_pos = np.array([self.wall_x, y, z])
            
            # Attempt to move robot to point with perpendicular tool
            print(f"\nAttempting point {i+1}/{len(sorted_points)}: {target_pos}")
            success = self.move_robot_to_point(target_pos, robot_type)
            
            if success:
                success_count += 1
                print(f"Point {i+1}/{len(sorted_points)}: Success")
                
                # Pause to visualize
                time.sleep(pause_time)
            else:
                print(f"Point {i+1}/{len(sorted_points)}: Failed")
        
        print(f"\nDemonstration completed: {success_count}/{len(sorted_points)} points reached successfully")
        
        # Reset robot to a neutral position
        if robot_type == "flat":
            neutral_pos = np.array([0.0, -1.57, 0.0, -1.57, 0.0, 0.0])
            for i, joint_id in enumerate(self.flat_ik_solver.joint_ids):
                self.data.qpos[joint_id] = neutral_pos[i]
        else:
            neutral_pos = np.array([np.pi/2, -1.57, 0.0, -1.57, 0.0, 0.0])
            for i, joint_id in enumerate(self.perp_ik_solver.joint_ids):
                self.data.qpos[joint_id] = neutral_pos[i]
        
        mujoco.mj_forward(self.model, self.data) 