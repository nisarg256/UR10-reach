import numpy as np
from scipy.spatial.transform import Rotation

def rotation_matrix_to_quaternion(R):
    """Convert a rotation matrix to a quaternion."""
    rotation = Rotation.from_matrix(R)
    return rotation.as_quat()

def quaternion_to_rotation_matrix(q):
    """Convert a quaternion to a rotation matrix."""
    rotation = Rotation.from_quat(q)
    return rotation.as_matrix()

def transform_to_matrix(position, orientation):
    """
    Convert position and orientation (quaternion) to a 4x4 transformation matrix.
    
    Args:
        position: [x, y, z] position vector
        orientation: [x, y, z, w] quaternion
        
    Returns:
        4x4 transformation matrix
    """
    R = quaternion_to_rotation_matrix(orientation)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = position
    return T

def check_perpendicular_to_wall(orientation, tolerance=0.1):
    """
    Check if the orientation is perpendicular to the wall.
    Assuming the wall is in the YZ plane (normal along X-axis).
    
    Args:
        orientation: [x, y, z, w] quaternion
        tolerance: cosine similarity tolerance (0 means exactly perpendicular)
        
    Returns:
        True if end effector is perpendicular to wall within tolerance
    """
    # Convert quaternion to rotation matrix
    R = quaternion_to_rotation_matrix(orientation)
    
    # Z-axis of the end effector
    z_axis = R[:, 2]
    
    # Wall normal (X-axis)
    wall_normal = np.array([1, 0, 0])
    
    # Cosine similarity between z_axis and wall_normal
    cosine_similarity = np.abs(np.dot(z_axis, wall_normal))
    
    # Check if the cosine similarity is close to 1 (vectors parallel or anti-parallel)
    return cosine_similarity > (1 - tolerance)

def calculate_workspace_points(joint_configs, forward_kinematics_fn):
    """
    Calculate workspace points from a set of joint configurations.
    
    Args:
        joint_configs: Array of joint configurations, shape (n, dof)
        forward_kinematics_fn: Function that takes joint config and returns end effector pose
        
    Returns:
        Array of end effector positions, shape (n, 3)
        Array of end effector orientations as quaternions, shape (n, 4)
    """
    n_samples = len(joint_configs)
    positions = np.zeros((n_samples, 3))
    orientations = np.zeros((n_samples, 4))
    
    for i, config in enumerate(joint_configs):
        pose = forward_kinematics_fn(config)
        positions[i] = pose[:3]
        orientations[i] = pose[3:]
        
    return positions, orientations 