import mujoco
import numpy as np
import os
import tempfile
from pathlib import Path
from scipy.spatial.transform import Rotation

from src.utils.kinematics import rotation_matrix_to_quaternion, transform_to_matrix

class UR10Simulator:
    """A MuJoCo simulator for the UR10 robot."""
    
    def __init__(self, urdf_path, mounting_config="flat"):
        """
        Initialize the UR10 simulator.
        
        Args:
            urdf_path: Path to the UR10 URDF file
            mounting_config: "flat" or "perpendicular"
        """
        self.urdf_path = urdf_path
        self.mounting_config = mounting_config
        
        # Load configuration
        from ..config import CONFIGURATIONS, SIM_TIMESTEP, GRAVITY
        self.position = CONFIGURATIONS[mounting_config]["position"]
        self.orientation = CONFIGURATIONS[mounting_config]["orientation"]
        self.timestep = SIM_TIMESTEP
        self.gravity = GRAVITY
        
        # Initialize MuJoCo model
        self._setup_mujoco_model()
    
    def _setup_mujoco_model(self):
        """Set up the MuJoCo model from the URDF."""
        # Create a temporary XML file for MuJoCo
        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
            temp_xml_path = f.name
        
        # Compile URDF to MuJoCo XML
        mujoco.mj_loadURDF(temp_xml_path, self.urdf_path, None)
        
        # Load the MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(temp_xml_path)
        
        # Set gravity
        self.model.opt.gravity = self.gravity
        
        # Set the robot base position and orientation
        self._set_robot_base_pose()
        
        # Initialize the data
        self.data = mujoco.MjData(self.model)
        
        # Clean up
        os.unlink(temp_xml_path)
    
    def _set_robot_base_pose(self):
        """Set the robot base position and orientation."""
        # Find the base body ID
        base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        
        if base_id >= 0:
            # Set the position
            self.model.body_pos[base_id] = self.position
            
            # Set the orientation (quaternion)
            self.model.body_quat[base_id] = self.orientation
    
    def reset(self, joint_positions=None):
        """
        Reset the simulation.
        
        Args:
            joint_positions: Optional initial joint positions
        """
        mujoco.mj_resetData(self.model, self.data)
        
        if joint_positions is not None:
            self.set_joint_positions(joint_positions)
        
        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)
    
    def step(self, control=None):
        """
        Step the simulation.
        
        Args:
            control: Optional control signal
        """
        if control is not None:
            self.data.ctrl[:] = control
        
        mujoco.mj_step(self.model, self.data)
    
    def set_joint_positions(self, joint_positions):
        """
        Set the joint positions.
        
        Args:
            joint_positions: Array of joint positions
        """
        # Find joint IDs and set positions
        for i, name in enumerate(["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]):
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name + "_joint")
            if joint_id >= 0:
                self.data.qpos[joint_id] = joint_positions[i]
        
        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)
    
    def get_end_effector_pose(self):
        """
        Get the end effector pose.
        
        Returns:
            Position (3) and orientation as quaternion (4)
        """
        # Find the end effector site ID
        ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        
        if ee_id < 0:
            # If no site is defined, use the last body (usually the end effector)
            ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "tool0")
        
        # Get the position and orientation matrix
        if ee_id >= 0:
            # For sites
            if mujoco.mjtObj.mjOBJ_SITE in [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SITE, ee_id)]:
                pos = self.data.site_xpos[ee_id].copy()
                mat = self.data.site_xmat[ee_id].reshape(3, 3).copy()
            # For bodies
            else:
                pos = self.data.body_xpos[ee_id].copy()
                mat = self.data.body_xmat[ee_id].reshape(3, 3).copy()
            
            # Convert rotation matrix to quaternion
            quat = rotation_matrix_to_quaternion(mat)
            
            return np.concatenate([pos, quat])
        
        return np.zeros(7)  # Default empty pose
    
    def sample_joint_positions(self, n_samples=1000):
        """
        Sample random joint positions within limits.
        
        Args:
            n_samples: Number of samples
            
        Returns:
            Array of joint positions, shape (n_samples, 6)
        """
        from ..config import UR10_JOINT_LIMITS
        
        joint_limits = np.array([
            UR10_JOINT_LIMITS["shoulder_pan"],
            UR10_JOINT_LIMITS["shoulder_lift"],
            UR10_JOINT_LIMITS["elbow"],
            UR10_JOINT_LIMITS["wrist_1"],
            UR10_JOINT_LIMITS["wrist_2"],
            UR10_JOINT_LIMITS["wrist_3"]
        ])
        
        # Sample uniformly within limits
        joint_positions = np.random.uniform(
            joint_limits[:, 0],
            joint_limits[:, 1],
            size=(n_samples, 6)
        )
        
        return joint_positions
    
    def compute_workspace(self, n_samples=1000):
        """
        Compute the workspace of the robot.
        
        Args:
            n_samples: Number of samples
            
        Returns:
            Dictionary with workspace points and orientations
        """
        # Sample random joint positions
        joint_positions = self.sample_joint_positions(n_samples)
        
        # Compute end effector poses
        positions = np.zeros((n_samples, 3))
        orientations = np.zeros((n_samples, 4))
        
        for i, joints in enumerate(joint_positions):
            self.reset(joints)
            pose = self.get_end_effector_pose()
            positions[i] = pose[:3]
            orientations[i] = pose[3:]
        
        return {
            "workspace_points": positions,
            "orientations": orientations,
            "joint_positions": joint_positions
        } 