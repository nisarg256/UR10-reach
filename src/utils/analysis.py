import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

def compute_workspace_volume(points):
    """
    Compute the volume of the workspace using a convex hull.
    
    Args:
        points: Array of 3D points, shape (n, 3)
        
    Returns:
        Volume of the convex hull
    """
    if len(points) < 4:
        return 0.0
    
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception as e:
        print(f"Error computing convex hull: {e}")
        return 0.0

def compute_reach_on_wall(positions, orientations, wall_position, wall_height, wall_width):
    """
    Compute the reach on a wall.
    
    Args:
        positions: Array of end effector positions, shape (n, 3)
        orientations: Array of end effector orientations as quaternions, shape (n, 4)
        wall_position: Position of the wall [x, y, z]
        wall_height: Height of the wall
        wall_width: Width of the wall
        
    Returns:
        Dictionary with reach statistics
    """
    from .kinematics import check_perpendicular_to_wall
    
    # Filter points that are close to the wall
    wall_x = wall_position[0]
    wall_distance_tolerance = 0.05  # 5cm tolerance
    
    wall_points = []
    perpendicular_points = []
    
    for pos, orient in zip(positions, orientations):
        # Check if the point is close to the wall
        if abs(pos[0] - wall_x) < wall_distance_tolerance:
            wall_points.append(pos)
            
            # Check if the orientation is perpendicular to the wall
            if check_perpendicular_to_wall(orient):
                perpendicular_points.append(pos)
    
    wall_points = np.array(wall_points) if wall_points else np.zeros((0, 3))
    perpendicular_points = np.array(perpendicular_points) if perpendicular_points else np.zeros((0, 3))
    
    # Compute reach statistics
    wall_coverage = len(wall_points) / len(positions) if len(positions) > 0 else 0
    perpendicular_coverage = len(perpendicular_points) / len(positions) if len(positions) > 0 else 0
    
    # Compute coverage area on the wall
    if len(perpendicular_points) >= 3:
        # Project points onto the YZ plane
        projected_points = perpendicular_points[:, 1:]
        try:
            hull = ConvexHull(projected_points)
            coverage_area = hull.volume  # In 2D, volume is actually area
        except Exception:
            coverage_area = 0
    else:
        coverage_area = 0
    
    return {
        "wall_points": wall_points,
        "perpendicular_points": perpendicular_points,
        "wall_coverage": wall_coverage,
        "perpendicular_coverage": perpendicular_coverage,
        "coverage_area": coverage_area
    }

def visualize_workspace(points, perpendicular_points=None, wall_position=None, wall_height=None, wall_width=None, title="Workspace Visualization"):
    """
    Visualize the workspace in 3D.
    
    Args:
        points: Array of 3D points, shape (n, 3)
        perpendicular_points: Optional array of points with perpendicular orientation
        wall_position: Optional position of the wall [x, y, z]
        wall_height: Optional height of the wall
        wall_width: Optional width of the wall
        title: Plot title
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot all workspace points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='blue', alpha=0.3, label='Workspace')
    
    # Plot perpendicular points if provided
    if perpendicular_points is not None and len(perpendicular_points) > 0:
        ax.scatter(perpendicular_points[:, 0], perpendicular_points[:, 1], perpendicular_points[:, 2], 
                  c='red', alpha=0.7, label='Perpendicular to Wall')
    
    # Add wall visualization if parameters are provided
    if all(param is not None for param in [wall_position, wall_height, wall_width]):
        x = wall_position[0]
        y_min, y_max = -wall_width/2, wall_width/2
        z_min, z_max = 0, wall_height
        
        y, z = np.meshgrid([y_min, y_max], [z_min, z_max])
        x = np.ones_like(y) * x
        
        ax.plot_surface(x, y, z, alpha=0.2, color='gray')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Add legend
    ax.legend()
    
    # Show equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    return fig

def compare_configurations(flat_results, perpendicular_results):
    """
    Compare the results of flat and perpendicular configurations.
    
    Args:
        flat_results: Results dictionary for flat configuration
        perpendicular_results: Results dictionary for perpendicular configuration
        
    Returns:
        Comparison dictionary
    """
    comparison = {
        "flat_workspace_volume": compute_workspace_volume(flat_results["workspace_points"]),
        "perpendicular_workspace_volume": compute_workspace_volume(perpendicular_results["workspace_points"]),
        "flat_wall_coverage": flat_results["wall_coverage"],
        "perpendicular_wall_coverage": perpendicular_results["wall_coverage"],
        "flat_perpendicular_coverage": flat_results["perpendicular_coverage"],
        "perpendicular_perpendicular_coverage": perpendicular_results["perpendicular_coverage"],
        "flat_coverage_area": flat_results["coverage_area"],
        "perpendicular_coverage_area": perpendicular_results["coverage_area"]
    }
    
    # Determine which configuration is better for wall coverage
    if comparison["flat_coverage_area"] > comparison["perpendicular_coverage_area"]:
        comparison["better_configuration"] = "flat"
    elif comparison["flat_coverage_area"] < comparison["perpendicular_coverage_area"]:
        comparison["better_configuration"] = "perpendicular"
    else:
        comparison["better_configuration"] = "equal"
    
    return comparison 