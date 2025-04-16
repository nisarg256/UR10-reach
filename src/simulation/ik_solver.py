#!/usr/bin/env python3
import numpy as np
from scipy.optimize import minimize
import mujoco
import time
import sys
from pathlib import Path

class UR10eIKSolver:
    """Inverse Kinematics solver for UR10e robot with perpendicularity constraint."""
    
    def __init__(self, model, data, robot_prefix="flat_"):
        """
        Initialize the IK solver.
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
            robot_prefix: Prefix for the robot joints ('flat_' or 'perp_')
        """
        self.model = model
        self.data = data
        self.prefix = robot_prefix
        self.is_perp_robot = "perp_" in robot_prefix
        
        # Get joint IDs for the robot
        self.joint_names = [
            f"{self.prefix}shoulder_pan_joint",
            f"{self.prefix}shoulder_lift_joint",
            f"{self.prefix}elbow_joint",
            f"{self.prefix}wrist_1_joint",
            f"{self.prefix}wrist_2_joint",
            f"{self.prefix}wrist_3_joint"
        ]
        
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.joint_names]
        
        # Get site ID for the tool tip
        self.tool_tip_name = f"{self.prefix}tool_tip"
        self.tool_tip_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.tool_tip_name)
        
        # Get robot base body ID to determine its orientation
        self.base_name = f"{self.prefix}robot_base"
        self.base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.base_name)
        
        # Also get the robot tool body for better perpendicularity control
        self.tool_name = f"{self.prefix}tool"
        self.tool_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.tool_name)
        
        # Get wall position
        self.wall_x = 1.2  # X coordinate of the wall
        
        # Get joint limits
        self.joint_limits = []
        for i, joint_id in enumerate(self.joint_ids):
            lower = self.model.jnt_range[joint_id][0]
            upper = self.model.jnt_range[joint_id][1]
            self.joint_limits.append((lower, upper))
            
        # Get base position and orientation - important for determining perpendicularity
        mujoco.mj_forward(self.model, self.data)
        self.base_pos = self.data.xpos[self.base_id].copy()
        self.base_ori = self.get_base_orientation()
        
        # Debug: Check tool orientation at initialization
        self.debug_tool_ori()
        
        # Print configuration info
        print(f"IK Solver initialized for {robot_prefix} robot")
        print(f"- Tool tip site: {self.tool_tip_name}")
        print(f"- Wall position at X = {self.wall_x}")
        print(f"- Base position: {self.base_pos}")
    
    def debug_tool_ori(self):
        """Debug tool orientation to help understand the coordinate frames."""
        tool_rot = self.data.xmat[self.tool_id].reshape(3, 3).copy()
        print(f"\nTool orientation for {self.prefix}:")
        print(f"X-axis: {tool_rot[:, 0]}")
        print(f"Y-axis: {tool_rot[:, 1]}") # This should point along tool axis now
        print(f"Z-axis: {tool_rot[:, 2]}")
        
        # Wall normal points in negative X direction
        wall_normal = np.array([-1.0, 0.0, 0.0])
        alignment = np.abs(np.dot(tool_rot[:, 1], wall_normal))
        print(f"Current alignment with wall normal: {alignment}")
        print(f"Is perpendicular to wall: {alignment > np.cos(0.2)}")
    
    def get_tool_pose(self, q=None):
        """
        Get the position and orientation of the tool tip.
        
        Args:
            q: Joint configuration (if None, use current state)
        
        Returns:
            pos: 3D position of the tool tip
            rot: 3×3 rotation matrix of tool
        """
        # Set joint positions if provided
        if q is not None:
            old_qpos = self.data.qpos.copy()
            for i, joint_id in enumerate(self.joint_ids):
                self.data.qpos[joint_id] = q[i]
            mujoco.mj_forward(self.model, self.data)
        
        # Get tool tip position
        pos = self.data.site_xpos[self.tool_tip_id].copy()
        
        # Get orientation (rotation matrix) from the tool body
        rot = self.data.xmat[self.tool_id].reshape(3, 3).copy()
        
        # Restore original joint positions if we changed them
        if q is not None:
            self.data.qpos[:] = old_qpos
            mujoco.mj_forward(self.model, self.data)
        
        return pos, rot
    
    def get_base_orientation(self):
        """Get the orientation of the robot base."""
        # Get the rotation matrix of the base
        base_rot = self.data.xmat[self.base_id].reshape(3, 3).copy()
        return base_rot
    
    def is_perpendicular_to_wall(self, tool_rot, tolerance=0.15):
        """
        Check if the tool is perpendicular to the wall.
        
        Args:
            tool_rot: 3×3 rotation matrix of the tool
            tolerance: Angular tolerance in radians
        
        Returns:
            bool: True if perpendicular within tolerance
        """
        # For the drywall finishing tool:
        # The tool's y-axis should be perpendicular to the wall (pointing toward the wall)
        # Wall normal points in negative X direction [-1,0,0]
        wall_normal = np.array([-1.0, 0.0, 0.0])
        
        # The tool's y-axis from its rotation matrix - this should be aligned with wall normal
        # for the tool to be perpendicular to the wall
        tool_y_axis = tool_rot[:, 1]
        
        # Calculate alignment (dot product should be close to 1 for good alignment)
        alignment = np.abs(np.dot(tool_y_axis, wall_normal))
        
        # Consider perpendicular if alignment is close to 1
        return alignment > np.cos(tolerance)
    
    def distance_to_wall(self, pos):
        """
        Calculate distance from tool tip to the wall.
        
        Args:
            pos: 3D position of tool tip
        
        Returns:
            float: Distance to wall
        """
        # Wall is at X = 1.2
        return abs(pos[0] - self.wall_x)
    
    def ik_cost_function(self, q, target_pos):
        """
        Cost function for IK optimization.
        
        Args:
            q: Joint configuration
            target_pos: Target position [x, y, z]
        
        Returns:
            float: Cost value (lower is better)
        """
        # Save current state
        old_qpos = self.data.qpos.copy()
        
        # Set joint positions
        for i, joint_id in enumerate(self.joint_ids):
            self.data.qpos[joint_id] = q[i]
        
        mujoco.mj_forward(self.model, self.data)
        
        # Get current tool tip position and orientation
        curr_pos = self.data.site_xpos[self.tool_tip_id].copy()
        curr_rot = self.data.xmat[self.tool_id].reshape(3, 3).copy()
        
        # Calculate position error
        pos_error = np.linalg.norm(curr_pos - target_pos)
        
        # Wall normal points in negative X direction
        wall_normal = np.array([-1.0, 0.0, 0.0])
        
        # The tool's y-axis from its rotation matrix
        tool_y_axis = curr_rot[:, 1]
        
        # Calculate perpendicularity error (1 - alignment)
        # Alignment should be close to 1 for perpendicularity
        alignment = np.abs(np.dot(tool_y_axis, wall_normal))
        perp_error = 1.0 - alignment  # Lower is better
        
        # Distance to wall error - we want tool tip to be exactly at the wall
        wall_error = self.distance_to_wall(curr_pos)
        
        # Calculate joint limit penalty
        limit_penalty = 0
        for i, (lower, upper) in enumerate(self.joint_limits):
            # Apply a soft penalty as we approach limits
            margin = 0.1  # radians
            if q[i] < lower + margin:
                limit_penalty += ((lower + margin) - q[i])**2
            elif q[i] > upper - margin:
                limit_penalty += (q[i] - (upper - margin))**2
        
        # Compute cost with increased weights for perpendicularity
        cost = (
            10.0 * pos_error +       # Position error 
            100.0 * perp_error +     # Perpendicularity is most critical - increased weight
            20.0 * wall_error +      # Wall contact is important
            1.0 * limit_penalty      # Avoid joint limits
        )
        
        # Restore original state
        self.data.qpos[:] = old_qpos
        mujoco.mj_forward(self.model, self.data)
        
        return cost
    
    def solve_ik(self, target_pos, initial_guess=None, max_iterations=300):
        """
        Solve inverse kinematics to reach target position with perpendicularity constraint.
        
        Args:
            target_pos: Target position [x, y, z]
            initial_guess: Initial joint configuration (if None, use current state)
            max_iterations: Maximum iterations for optimizer
        
        Returns:
            q: Joint configuration or None if no solution found
            success: True if successful
        """
        # Check that the target is at the wall
        if abs(target_pos[0] - self.wall_x) > 0.01:
            # Project the target onto the wall
            target_pos_wall = target_pos.copy()
            target_pos_wall[0] = self.wall_x
            target_pos = target_pos_wall
        
        # Use current configuration as initial guess if not provided
        if initial_guess is None:
            initial_guess = np.array([self.data.qpos[joint_id] for joint_id in self.joint_ids])
        
        # Define bounds for optimization (joint limits)
        bounds = self.joint_limits
        
        # Multiple optimization attempts with different initial guesses
        best_result = None
        best_cost = float('inf')
        
        # First attempt with provided/current configuration
        result = minimize(
            self.ik_cost_function,
            initial_guess,
            args=(target_pos,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iterations, 'ftol': 1e-6}
        )
        
        if result.success and result.fun < best_cost:
            best_result = result
            best_cost = result.fun
        
        # Try with specific configurations for each robot type
        if self.is_perp_robot:
            # For the perpendicular robot
            base_configs = [
                # Home position
                np.array([np.pi/2, -1.57, 0.0, -1.57, 0.0, 0.0]),
                # Different shoulder lift and elbow combinations
                np.array([np.pi/2, -0.5, 0.5, -1.57, 0.0, 0.0]),
                np.array([np.pi/2, -1.0, 1.0, -1.57, 0.0, 0.0]),
                np.array([np.pi/2, -2.0, 1.5, -1.57, 0.0, 0.0])
            ]
        else:
            # For the flat robot
            base_configs = [
                # Home position
                np.array([0.0, -1.57, 0.0, -1.57, 0.0, 0.0]),
                # Different shoulder lift and elbow combinations
                np.array([0.0, -0.5, 0.5, -1.57, 0.0, 0.0]),
                np.array([0.0, -1.0, 1.0, -1.57, 0.0, 0.0]),
                np.array([0.0, -2.0, 1.5, -1.57, 0.0, 0.0])
            ]
        
        # Calculate angle to target from base position
        y_diff = target_pos[1] - self.base_pos[1]
        x_diff = target_pos[0] - self.base_pos[0]
        target_angle = np.arctan2(y_diff, x_diff)
        
        # Add configurations with shoulder pan oriented toward target
        for base_config in base_configs.copy():
            if self.is_perp_robot:
                # For perp robot, adjust shoulder pan from π/2
                pan_angle = np.pi/2 + np.arctan2(y_diff, abs(x_diff))
                config = base_config.copy()
                config[0] = pan_angle
                base_configs.append(config)
            else:
                # For flat robot, use direct angle
                config = base_config.copy()
                config[0] = target_angle
                base_configs.append(config)
        
        # Try all specified configurations
        for config in base_configs:
            result = minimize(
                self.ik_cost_function,
                config,
                args=(target_pos,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iterations, 'ftol': 1e-6}
            )
            
            if result.success and result.fun < best_cost:
                best_result = result
                best_cost = result.fun
        
        # Try with different wrist configurations to help with perpendicularity
        wrist_configs = []
        if best_result is not None:
            base_solution = best_result.x.copy()
            
            # Try variations of wrist joints
            for wrist1 in [-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2]:
                for wrist2 in [-np.pi/4, 0, np.pi/4]:
                    for wrist3 in [-np.pi/4, 0, np.pi/4]:
                        config = base_solution.copy()
                        config[3] = wrist1  # wrist1
                        config[4] = wrist2  # wrist2
                        config[5] = wrist3  # wrist3
                        wrist_configs.append(config)
            
            # Try these wrist configurations
            for config in wrist_configs:
                result = minimize(
                    self.ik_cost_function,
                    config,
                    args=(target_pos,),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': max_iterations, 'ftol': 1e-6}
                )
                
                if result.success and result.fun < best_cost:
                    best_result = result
                    best_cost = result.fun
        
        # Check if we found a good solution
        if best_result is not None:
            q = best_result.x
            
            # Verify solution
            old_qpos = self.data.qpos.copy()
            for i, joint_id in enumerate(self.joint_ids):
                self.data.qpos[joint_id] = q[i]
            
            mujoco.mj_forward(self.model, self.data)
            
            pos, rot = self.get_tool_pose()
            pos_error = np.linalg.norm(pos - target_pos)
            is_perp = self.is_perpendicular_to_wall(rot)
            
            # Restore original state
            self.data.qpos[:] = old_qpos
            mujoco.mj_forward(self.model, self.data)
            
            # Only succeed if both position accuracy and perpendicularity are achieved
            if pos_error < 0.05 and is_perp:
                return q, True
            elif pos_error < 0.1 and is_perp:  # Slightly relaxed position criteria
                return q, True
        
        return None, False
    
    def is_point_reachable(self, target_pos, num_attempts=5):
        """
        Check if a point is reachable with the perpendicularity constraint.
        
        Args:
            target_pos: Target position [x, y, z]
            num_attempts: Number of IK attempts with different initial guesses
        
        Returns:
            bool: True if reachable
            q: Joint configuration if reachable, None otherwise
        """
        # Try to solve IK directly
        q, success = self.solve_ik(target_pos)
        if success:
            return True, q
        
        # Try with random initial guesses
        for _ in range(num_attempts - 1):
            # Generate random initial guess within joint limits
            initial_guess = np.array([
                np.random.uniform(lower + 0.2, upper - 0.2) 
                for lower, upper in self.joint_limits
            ])
            
            q, success = self.solve_ik(target_pos, initial_guess)
            if success:
                return True, q
        
        return False, None 