#!/usr/bin/env python3
import os
import sys
import time
import argparse
import numpy as np
import mujoco
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Simple MuJoCo Viewer for UR10 models")
    parser.add_argument("--model", type=str, default="src/models/reach_comparison.xml", 
                       help="Path to MuJoCo model XML file")
    parser.add_argument("--animate", action="store_true", help="Animate the simulation")
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
    
    # Reset data to initial state
    mujoco.mj_resetData(model, data)
    
    # Print some model information
    print(f"Model loaded: {model.nq} DOFs, {model.nbody} bodies")
    print(f"Joint names: {[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(min(10, model.njnt))]}")
    
    # Initialize visualization data structures
    scene = mujoco.MjvScene(model, maxgeom=10000)
    camera = mujoco.MjvCamera()
    option = mujoco.MjvOption()
    
    # Initialize GLFW
    from mujoco.glfw import glfw
    if not glfw.init():
        print("Could not initialize GLFW")
        sys.exit(1)
    
    # Create window
    window = glfw.create_window(1200, 900, "UR10 Model Viewer", None, None)
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
    
    # Main loop
    print("\nViewer controls:")
    print("  ESC: Exit viewer")
    print("  R: Reset camera view")
    print("  Left mouse drag: Rotate camera")
    print("  Right mouse drag: Pan camera (horizontal/vertical)")
    print("  Shift + Right mouse drag: Pan camera (horizontal/depth)")
    print("  Mouse wheel: Zoom camera")
    
    print("\nStarting viewer...")
    try:
        while not glfw.window_should_close(window):
            time_start = time.time()
            
            # Update modifier tracking
            mods_shift = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or \
                        glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
            
            # Step simulation if animated
            if args.animate:
                mujoco.mj_step(model, data)
            
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
            
            # Check for ESC press
            if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
                break
            
            # Control framerate
            if args.animate:
                time_consumed = time.time() - time_start
                time_wait = max(0, 1.0/60.0 - time_consumed)
                time.sleep(time_wait)
    
    except KeyboardInterrupt:
        print("\nViewer interrupted by user")
    
    finally:
        # Clean up
        glfw.terminate()
        print("\nViewer closed")

if __name__ == "__main__":
    main() 