#!/usr/bin/env python3
import os
import sys
import time
import argparse
import numpy as np
import mujoco
import mujoco.glfw.glfw as glfw
from pathlib import Path

class MujocoViewer:
    """
    Custom MuJoCo viewer using the low-level GLFW API.
    """
    def __init__(self, model, data, title="MuJoCo Viewer"):
        """
        Initialize the MuJoCo viewer.
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
            title: Window title
        """
        self.model = model
        self.data = data
        self.title = title
        
        # GLFW initialization
        if not glfw.init():
            raise Exception("GLFW initialization failed")
        
        # Configure window
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
        glfw.window_hint(glfw.SAMPLES, 4)  # anti-aliasing
        
        # Create window
        width, height = 1200, 900
        self.window = glfw.create_window(width, height, title, None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("Window creation failed")
        
        # Make context current
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)  # vsync
        
        # Initialize MuJoCo visualization structures
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.scn = mujoco.MjvScene(model, maxgeom=10000)
        self.con = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        
        # Initialize the camera
        mujoco.mjv_defaultCamera(self.cam)
        mujoco.mjv_defaultOption(self.opt)
        
        # Set camera position for a good view
        self.cam.lookat[0] = 0.0  # x-position to look at
        self.cam.lookat[1] = 0.0  # y-position to look at
        self.cam.lookat[2] = 1.0  # z-position to look at
        self.cam.distance = 4.0    # distance from focal point
        self.cam.azimuth = 90      # rotation around z-axis (in degrees)
        self.cam.elevation = -20   # elevation angle (in degrees)
        
        # Register keyboard callback
        glfw.set_key_callback(self.window, self._keyboard_callback)
        
        # Register mouse callbacks
        glfw.set_cursor_pos_callback(self.window, self._mouse_move_callback)
        glfw.set_mouse_button_callback(self.window, self._mouse_button_callback)
        glfw.set_scroll_callback(self.window, self._scroll_callback)
        
        # Mouse control variables
        self.last_x = 0
        self.last_y = 0
        self.button_left = False
        self.button_right = False
        self.button_middle = False
        
    def _keyboard_callback(self, window, key, scancode, action, mods):
        """Keyboard callback for camera control."""
        if action == glfw.PRESS and key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
            
        # Reset camera with R key
        if action == glfw.PRESS and key == glfw.KEY_R:
            mujoco.mjv_defaultCamera(self.cam)
            self.cam.lookat[0] = 0.0
            self.cam.lookat[1] = 0.0
            self.cam.lookat[2] = 1.0
            self.cam.distance = 4.0
            self.cam.azimuth = 90
            self.cam.elevation = -20
    
    def _mouse_button_callback(self, window, button, act, mods):
        """Mouse button callback for camera control."""
        # Update button states
        self.button_left = (glfw.MOUSE_BUTTON_LEFT == button and glfw.PRESS == act)
        self.button_middle = (glfw.MOUSE_BUTTON_MIDDLE == button and glfw.PRESS == act)
        self.button_right = (glfw.MOUSE_BUTTON_RIGHT == button and glfw.PRESS == act)
        
        # Save mouse position when button is pressed
        if self.button_left or self.button_middle or self.button_right:
            self.last_x, self.last_y = glfw.get_cursor_pos(window)
    
    def _mouse_move_callback(self, window, x_pos, y_pos):
        """Mouse move callback for camera control."""
        # Compute mouse displacement
        dx = x_pos - self.last_x
        dy = y_pos - self.last_y
        
        # Update camera based on mouse movement
        if self.button_right:
            # Rotate camera
            self.cam.azimuth += 0.5 * dx
            self.cam.elevation += 0.5 * dy
            self.cam.elevation = max(min(self.cam.elevation, 90), -90)
            
        elif self.button_left:
            # Translate camera target
            forward = np.array([np.cos(np.deg2rad(self.cam.azimuth)) * np.cos(np.deg2rad(self.cam.elevation)),
                               np.sin(np.deg2rad(self.cam.azimuth)) * np.cos(np.deg2rad(self.cam.elevation)),
                               np.sin(np.deg2rad(self.cam.elevation))])
            right = np.array([-np.sin(np.deg2rad(self.cam.azimuth)), 
                             np.cos(np.deg2rad(self.cam.azimuth)),
                             0])
            up = np.cross(right, forward)
            
            # Scale translation based on camera distance
            scale = 0.01 * self.cam.distance
            self.cam.lookat[0] += scale * (-right[0] * dx + up[0] * dy)
            self.cam.lookat[1] += scale * (-right[1] * dx + up[1] * dy)
            self.cam.lookat[2] += scale * (-right[2] * dx + up[2] * dy)
            
        elif self.button_middle:
            # Adjust camera distance
            self.cam.distance *= 1.0 - 0.01 * dy
            
        # Save mouse position
        self.last_x = x_pos
        self.last_y = y_pos
    
    def _scroll_callback(self, window, x_offset, y_offset):
        """Scroll callback for camera zooming."""
        self.cam.distance *= 1.0 - 0.05 * y_offset
        
    def render(self):
        """Render the scene."""
        # Setup the drawing viewport
        viewport_width, viewport_height = glfw.get_framebuffer_size(self.window)
        viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)
        
        # Update scene
        mujoco.mjv_updateScene(
            self.model, self.data, self.opt, None, self.cam,
            mujoco.mjtCatBit.mjCAT_ALL.value, self.scn
        )
        
        # Render scene
        mujoco.mjr_render(viewport, self.scn, self.con)
        
        # Swap buffers
        glfw.swap_buffers(self.window)
        
        # Process events
        glfw.poll_events()
        
    def is_running(self):
        """Check if the viewer is still running."""
        return not glfw.window_should_close(self.window)
    
    def close(self):
        """Close the viewer."""
        glfw.terminate()

def main():
    parser = argparse.ArgumentParser(description="MuJoCo Viewer for UR10 model")
    parser.add_argument("--model", type=str, default="src/models/reach_comparison.xml", 
                        help="Path to MuJoCo model XML file")
    parser.add_argument("--animate", action="store_true", help="Animate the model")
    parser.add_argument("--fps", type=int, default=60, help="Frames per second")
    args = parser.parse_args()
    
    # Load model
    model_path = Path(args.model)
    print(f"Loading model from {model_path}")
    
    try:
        model = mujoco.MjModel.from_xml_path(str(model_path))
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Reset the data to a valid state
    mujoco.mj_resetData(model, data)
    
    # Print model information
    print(f"Model loaded: {model.nq} DOFs, {model.nbody} bodies")
    print(f"Joint names: {[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(min(10, model.njnt))]}")
    
    # Create viewer
    print("Initializing viewer...")
    title = "UR10 Reach Analysis"
    viewer = MujocoViewer(model, data, title)
    
    # Set target FPS
    fps = args.fps
    step_time = 1.0 / fps
    
    print("\nControls:")
    print("  Mouse right-drag: Rotate camera")
    print("  Mouse left-drag: Move camera target")
    print("  Mouse middle-drag/scroll: Zoom")
    print("  R: Reset camera view")
    print("  ESC: Quit")
    
    # Main rendering loop
    print("\nStarting viewer...")
    try:
        last_update = time.time()
        while viewer.is_running():
            current_time = time.time()
            
            # Animate if requested
            if args.animate and current_time - last_update >= step_time:
                # Step physics
                mujoco.mj_step(model, data)
                last_update = current_time
                
                # Report position of end effectors
                for ee_name in ["flat_wrist_3_link", "perp_wrist_3_link"]:
                    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_name)
                    if body_id >= 0:
                        pos = data.xpos[body_id]
                        print(f"{ee_name} position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]", end='\r')
            
            # Render and process events
            viewer.render()
            
            # Limit refresh rate
            time_diff = step_time - (time.time() - current_time)
            if time_diff > 0:
                time.sleep(time_diff)
    
    except KeyboardInterrupt:
        print("\nViewer interrupted by user")
    finally:
        viewer.close()
        print("\nViewer closed")

if __name__ == "__main__":
    main() 