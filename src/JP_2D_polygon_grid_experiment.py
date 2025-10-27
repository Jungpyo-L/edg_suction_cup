#!/usr/bin/env python

# Authors: Jungpyo Lee
# Create: Oct.21.2025
# Last update: Oct.21.2025
# Description: This script is used to run polygon grid experiments for 2D haptic search.
#              It tests 12 polygons (3x4 grid) with different controllers and analyzes
#              success rates, path lengths, and iterations across the grid.
#              
#              Features:
#              - 12 polygon sequential testing: (0,0) → (1,0) → ... → (3,2)
#              - Random initial poses within 1.5-2.5cm range from disengagePose
#              - Trial validation: retry if suction succeeds at initial position
#              - Multiple controller testing: greedy, RL yaw_momentum, etc.
#              - Comprehensive data analysis and comparison

# imports
try:
  import rospy
  import tf
  ros_enabled = True
except:
  print('Couldn\'t import ROS.  I assume you\'re running this on your laptop')
  ros_enabled = False

import os, sys
import string
from helperFunction.utils import rotation_from_quaternion, create_transform_matrix, quaternion_from_matrix, normalize, hat

from datetime import datetime
import pandas as pd
import re
import subprocess
import numpy as np
import copy
import time
import scipy
import pickle
import shutil
from scipy.io import savemat
from scipy.spatial.transform import Rotation as sciRot

from netft_utils.srv import *
from suction_cup.srv import *
from std_msgs.msg import String
from std_msgs.msg import Int8
import geometry_msgs.msg
import tf
import cv2
from scipy import signal

from math import pi, cos, sin, floor

from helperFunction.FT_callback_helper import FT_CallbackHelp
from helperFunction.fileSaveHelper import fileSaveHelp
from helperFunction.rtde_helper import rtdeHelp
from helperFunction.hapticSearch2D import hapticSearch2DHelp
from helperFunction.SuctionP_callback_helper import P_CallbackHelp
from helperFunction.RL_controller_helper import RLControllerHelper, create_rl_controller
from helperFunction.utils import hat, create_transform_matrix

# All functions are self-contained in this file


def get_disEngagePosition(primitives):
    """
    Get the disengage position (safe position above the object) based on primitive type.
    This position is where the robot returns after each experiment.
    
    Args:
        primitives: Type of primitive shape ('polygons')
    
    Returns:
        List of [x, y, z] coordinates in meters for the disengage position
    """
    if primitives == 'polygons':
        disEngagePosition = [0.465, -.190, 0.013 +0.0015]  # Base position for polygon (0,0)
    else:
        # Default fallback for unknown primitives
        print(f"Warning: Unknown primitives '{primitives}', using polygon position")
        disEngagePosition = [0.443, -.038, 0.013]
    return disEngagePosition


def convert_yawAngle(yaw_radian):
    """
    Convert yaw angle from radians to degrees with coordinate transformation.
    
    Args:
        yaw_radian: Yaw angle in radians
    
    Returns:
        Converted yaw angle in degrees
    """
    yaw_angle = yaw_radian*180/pi
    yaw_angle = -yaw_angle - 90
    return yaw_angle


def normalize_yaw_angle(yaw_degrees):
    """
    Normalize yaw angle to [-180, 180] degrees range.
    
    Args:
        yaw_degrees: Yaw angle in degrees
    
    Returns:
        Normalized yaw angle in [-180, 180] degrees
    """
    while yaw_degrees > 180:
        yaw_degrees -= 360
    while yaw_degrees < -180:
        yaw_degrees += 360
    return yaw_degrees


def get_reverse_yaw_rotation(initial_yaw, current_yaw):
    """
    Calculate the reverse rotation to return to initial yaw from current yaw.
    This reverses the rotation that happened during haptic search.
    
    Args:
        initial_yaw: Initial yaw angle before haptic search (degrees)
        current_yaw: Current yaw angle after haptic search (degrees)
    
    Returns:
        Reverse rotation angle in degrees (opposite of haptic search rotation)
    """
    # Normalize both angles to [-180, 180]
    initial_yaw = normalize_yaw_angle(initial_yaw)
    current_yaw = normalize_yaw_angle(current_yaw)
    
    # Calculate the rotation that happened during haptic search
    haptic_rotation = current_yaw - initial_yaw
    
    # Handle 360-degree boundary crossing
    if haptic_rotation > 180:
        haptic_rotation -= 360
    elif haptic_rotation < -180:
        haptic_rotation += 360
    
    # Return the reverse rotation (opposite direction)
    reverse_rotation = -haptic_rotation
    
    return reverse_rotation


def process_pressure_data(P_array, search_help):
    """
    Apply consistent pressure processing (sign flipping and thresholding) 
    as used by the greedy controller.
    
    Args:
        P_array: Raw pressure array
        search_help: hapticSearch2DHelp instance with p_reverse and dP_threshold
        
    Returns:
        Processed pressure array
    """
    P_processed = P_array.copy()
    if not search_help.p_reverse:
        P_processed = [-P for P in P_processed]
    # Apply thresholding
    th = search_help.dP_threshold
    P_processed = [P if P > th else 0 for P in P_processed]
    return P_processed


def get_matching_heuristic_controller(rl_controller_name):
    """
    Get the matching heuristic controller for a given RL controller.
    
    Args:
        rl_controller_name: Name of the RL controller (e.g., 'rl_hyaw_momentum', 'rl_hgreedy')
    
    Returns:
        String name of the matching heuristic controller
    """
    # Remove 'rl_' prefix to get the base controller type
    base_name = rl_controller_name.replace('rl_', '')
    
    # Remove 'h' prefix if present (for models like 'hgreedy', 'hyaw', etc.)
    if base_name.startswith('h'):
        base_name = base_name[1:]  # Remove 'h' prefix
    
    # Map RL controller names to heuristic controller names
    controller_mapping = {
        'greedy': 'greedy',
        'yaw': 'yaw', 
        'momentum': 'momentum',
        'yaw_momentum': 'yaw_momentum',
        'momentum_yaw': 'yaw_momentum'
    }
    
    # Return the matching heuristic controller, default to 'greedy' if not found
    return controller_mapping.get(base_name, 'greedy')


def check_pose_safety(rtde_help, pose):
    """
    Check if a pose would violate joint limits before execution.
    
    Args:
        rtde_help: RTDE helper for pose validation
        pose: PoseStamped object to validate
        
    Returns:
        bool: True if pose is safe, False if it would violate joint limits
    """
    try:
        return rtde_help.validatePose(pose)
    except Exception as e:
        print(f"Pose safety check error: {e}")
        return False


def reset_robot_to_safe_orientation(rtde_help, target_position, safe_yaw_deg=45.0):
    """
    Reset robot to a safe orientation when it gets stuck in extreme poses.
    
    Args:
        rtde_help: RTDE helper for robot control
        target_position: Target position [x, y, z]
        safe_yaw_deg: Safe yaw angle in degrees (default: 45°)
        
    Returns:
        bool: True if reset successful, False otherwise
    """
    try:
        print(f"  Resetting robot to safe orientation (yaw={safe_yaw_deg}°)...")
        
        # Create safe pose with default orientation
        safe_yaw_rad = safe_yaw_deg * np.pi / 180
        setOrientation = tf.transformations.quaternion_from_euler(safe_yaw_rad, np.pi, 0, 'szxy')
        safe_pose = rtde_help.getPoseObj(target_position, setOrientation)
        
        # Move to safe pose
        rtde_help.goToPose_2Dhaptic(safe_pose)
        rospy.sleep(1.0)
        
        print(f"  Robot reset to safe orientation")
        return True
        
    except Exception as e:
        print(f"  Error resetting robot orientation: {e}")
        return False


def run_haptic_search_loop(args, rtde_help, search_help, P_help, targetPWM_Pub, 
                          dataLoggerEnable, file_help, disEngagePose, rl_controllers_dict, 
                          experiment_num=None, random_yaw=None, polygon_center_pose=None):
    """
    Run the main haptic search loop with hopping motion for polygon experiments.
    
    This is a simplified version specifically for polygon grid experiments.
    It handles:
    1. Starting data logging and vacuum pump
    2. Moving to engage position (5mm below disengage position)
    3. Running the haptic search control loop until success or max iterations
    4. Stopping data logging and vacuum pump
    
    Args:
        args: Command line arguments containing experiment parameters
        rtde_help: Helper for robot control via RTDE interface
        search_help: Helper for 2D haptic search computations
        P_help: Helper for pressure sensor data
        targetPWM_Pub: ROS publisher for vacuum pump PWM control
        dataLoggerEnable: ROS service proxy for enabling/disabling data logging
        file_help: Helper for saving experiment data
        disEngagePose: Safe position above the object (starting pose)
        rl_controllers_dict: Dictionary of RL controllers {controller_name: rl_controller}
        experiment_num: Optional experiment number (for repeated experiments)
        random_yaw: Optional random yaw value (for repeated experiments)
    
    Returns:
        Dictionary containing:
          - success: Boolean indicating if suction was achieved
          - iterations: Number of control iterations executed
          - elapsed_time: Total time for the experiment
          - final_yaw: Final yaw angle at end of experiment
          - path_length_2d: Total 2D path length (XY plane) in meters
          - path_length_3d: Total 3D path length (XYZ) in meters
    """
    
    # ==================== INITIALIZATION ====================
    DUTYCYCLE_100 = 100  # Full vacuum power
    DUTYCYCLE_0 = 0      # No vacuum
    P_vac = search_help.P_vac  # Vacuum pressure threshold for success
    max_steps = args.max_iterations  # Maximum iterations before giving up
    
    # ==================== START EXPERIMENT ====================
    targetPWM_Pub.publish(DUTYCYCLE_100)
    P_help.startSampling()      
    rospy.sleep(1)
    P_help.setNowAsOffset()
    dataLoggerEnable(True)

    # ==================== MOVE TO ENGAGE POSITION ====================
    rtde_help.goToPose_2Dhaptic(disEngagePose)
    rospy.sleep(0.2)
    
    # Move down to engage position (5mm below)
    engagePosition = copy.deepcopy(disEngagePose.pose.position)
    engagePosition.z = engagePosition.z - 5e-3
    engagePose = rtde_help.getPoseObj(engagePosition, disEngagePose.pose.orientation)
    rtde_help.goToPose_2Dhaptic(engagePose)
    
    # ==================== CAPTURE INITIAL POSE ====================
    # Capture the initial pose (disengage pose) for analysis
    initial_pose = rtde_help.getCurrentPose()
    args.initial_pose_position = [initial_pose.pose.position.x, initial_pose.pose.position.y, initial_pose.pose.position.z]
    args.initial_pose_orientation = [initial_pose.pose.orientation.x, initial_pose.pose.orientation.y, initial_pose.pose.orientation.z, initial_pose.pose.orientation.w]
    args.initial_pose_timestamp = initial_pose.header.stamp.to_sec()
    
    # ==================== INITIALIZE CONTROL LOOP VARIABLES ====================
    suctionSuccessFlag = False
    startTime = time.time()
    iteration_count = 1
    pose_diff = 0
    orientation_error = [0, 0, 0]
    
    # Reset yaw tracking for new experiment
    search_help.reset_yaw_tracking()
    
    # Note: Boundary checking is handled by polygon boundary check after trial completion
    
    # Path length tracking
    path_length_2d = 0.0
    path_length_3d = 0.0
    prev_position = None
    
    # Get the appropriate RL controller for this controller (None if heuristic)
    rl_controller = rl_controllers_dict.get(args.controller, None)
    
    # Reset RL controller history if using RL-based control
    if rl_controller and rl_controller.is_model_loaded():
        rl_controller.reset_history()
    
    # ==================== MAIN HAPTIC SEARCH CONTROL LOOP ====================
    while not suctionSuccessFlag:
        iteration_start_time = time.time()
        
        # ===== CHECK SUCCESS CONDITION =====
        P_array = P_help.four_pressure
        reached_vacuum = all(np.array(P_array) < P_vac)
        
        
        if reached_vacuum:
            suctionSuccessFlag = True
            args.elapsedTime = time.time() - startTime
            args.iteration_count = iteration_count
            
            # ==================== CAPTURE FINAL SUCCESS POSE ====================
            # Capture the final pose when success is achieved
            final_pose = rtde_help.getCurrentPose()
            args.final_pose_position = [final_pose.pose.position.x, final_pose.pose.position.y, final_pose.pose.position.z]
            args.final_pose_orientation = [final_pose.pose.orientation.x, final_pose.pose.orientation.y, final_pose.pose.orientation.z, final_pose.pose.orientation.w]
            args.final_pose_timestamp = final_pose.header.stamp.to_sec()
            
            if experiment_num:
                print(f"Experiment {experiment_num}: SUCCESS at iteration {iteration_count}")
            else:
                print(f"Suction engage succeeded with controller {args.controller} at iteration {iteration_count}")
            rtde_help.stopAtCurrPoseAdaptive()
            rospy.sleep(0.5)
            targetPWM_Pub.publish(DUTYCYCLE_0)
            rospy.sleep(0.5)
            break
        
        # ===== CHECK FAILURE CONDITION =====
        elif iteration_count >= max_steps:
            args.timeOverFlag = True
            args.elapsedTime = time.time() - startTime
            suctionSuccessFlag = False
            
            # ==================== CAPTURE FINAL FAILURE POSE ====================
            # Capture the final pose even when experiment fails
            final_pose = rtde_help.getCurrentPose()
            args.final_pose_position = [final_pose.pose.position.x, final_pose.pose.position.y, final_pose.pose.position.z]
            args.final_pose_orientation = [final_pose.pose.orientation.x, final_pose.pose.orientation.y, final_pose.pose.orientation.z, final_pose.pose.orientation.w]
            args.final_pose_timestamp = final_pose.header.stamp.to_sec()
            
            if experiment_num:
                print(f"Experiment {experiment_num}: FAILED - Max iterations reached")
            else:
                print(f"Suction controller failed! Reached maximum iterations ({max_steps})")
            rtde_help.stopAtCurrPoseAdaptive()
            targetPWM_Pub.publish(DUTYCYCLE_0)
            rospy.sleep(0.1)
            break

        # ===== GET CURRENT STATE =====
        measuredCurrPose = rtde_help.getCurrentPose()
        T_curr = search_help.get_Tmat_from_Pose(measuredCurrPose)
        
        # ===== CHECK BOUNDARY CONDITION =====
        if polygon_center_pose is not None:
            # Check if robot has moved too far from polygon center
            current_x = measuredCurrPose.pose.position.x
            current_y = measuredCurrPose.pose.position.y
            center_x = polygon_center_pose.pose.position.x
            center_y = polygon_center_pose.pose.position.y
            
            dx = current_x - center_x
            dy = current_y - center_y
            displacement = np.sqrt(dx**2 + dy**2)
            boundary_limit = 25e-3  # 25mm boundary limit from polygon center
            
            if displacement > boundary_limit:
                args.timeOverFlag = True
                args.elapsedTime = time.time() - startTime
                suctionSuccessFlag = False
                
                # ==================== CAPTURE FINAL BOUNDARY FAILURE POSE ====================
                # Capture the final pose when boundary is exceeded
                final_pose = rtde_help.getCurrentPose()
                args.final_pose_position = [final_pose.pose.position.x, final_pose.pose.position.y, final_pose.pose.position.z]
                args.final_pose_orientation = [final_pose.pose.orientation.x, final_pose.pose.orientation.y, final_pose.pose.orientation.z, final_pose.pose.orientation.w]
                args.final_pose_timestamp = final_pose.header.stamp.to_sec()
                
                if experiment_num:
                    print(f"Experiment {experiment_num}: FAILED - Boundary exceeded ({displacement*1000:.1f}mm > {boundary_limit*1000:.1f}mm)")
                else:
                    print(f"Suction controller failed! Boundary exceeded ({displacement*1000:.1f}mm > {boundary_limit*1000:.1f}mm)")
                rtde_help.stopAtCurrPoseAdaptive()
                targetPWM_Pub.publish(DUTYCYCLE_0)
                rospy.sleep(0.1)
                break
            
        yaw_angle = convert_yawAngle(search_help.get_yawRotation_from_T(T_curr))

        # ===== COMPUTE CONTROL ACTION =====
        if rl_controller and rl_controller.is_model_loaded():
            # RL-BASED CONTROL
            try:
                # Apply same pressure processing as greedy controller
                P_processed = process_pressure_data(P_array, search_help)
                
                lateral_vel, yaw_vel, debug_info = rl_controller.compute_action(P_processed, yaw_angle, return_debug=True)
                
                # Scale RL output from mm (model units) to m (robot units)
                mm_to_m = 1.0e-3
                dx_lat = lateral_vel[0] * mm_to_m
                dy_lat = lateral_vel[1] * mm_to_m
                
                # Apply same coordinate transformation as greedy controller
                # (swap x,y and negate both to match coordinate system)
                T_later = search_help.get_Tmat_TranlateInBodyF([-dy_lat, -dx_lat, 0.0])
                
                if abs(yaw_vel) > 1e-6:
                    d_yaw_rad = yaw_vel * np.pi / 180.0
                    rot_axis = np.array([0, 0, -1])
                    omega_hat = hat(rot_axis)
                    Rw = scipy.linalg.expm(d_yaw_rad * omega_hat)
                    T_yaw = create_transform_matrix(Rw, [0, 0, 0])
                else:
                    T_yaw = np.eye(4)
                
                T_align = np.eye(4)
                
            except Exception as e:
                print(f"RL controller failed: {e}, falling back to matching heuristic")
                # Use matching heuristic controller based on RL controller type
                # Use raw pressure data - heuristic controllers do their own pressure processing
                heuristic_controller = get_matching_heuristic_controller(args.controller)
                T_later, T_yaw, T_align = search_help.get_Tmats_from_controller(P_array, heuristic_controller)
        else:
            # HEURISTIC CONTROL
            # Use raw pressure data - heuristic controllers do their own pressure processing
            # Use proper controller name (convert RL name to heuristic name if needed)
            if args.controller.startswith('rl_'):
                heuristic_controller = get_matching_heuristic_controller(args.controller)
            else:
                heuristic_controller = args.controller
            T_later, T_yaw, T_align = search_help.get_Tmats_from_controller(P_array, heuristic_controller)
        
        # Combine transformations
        T_move = T_later @ T_yaw @ T_align

        # ===== EXECUTE HOPPING MOTION =====
        measuredCurrPose = rtde_help.getCurrentPose()
        currPose = search_help.get_PoseStamped_from_T_initPose(T_move, measuredCurrPose)
        
        # Apply orientation compensation
        q = currPose.pose.orientation
        curr_euler = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w], 'szxy')
        compensated_euler = [curr_euler[0] - orientation_error[0],
                            curr_euler[1] - orientation_error[1],
                            curr_euler[2] - orientation_error[2]]
        compensated_quat = tf.transformations.quaternion_from_euler(compensated_euler[0], compensated_euler[1], compensated_euler[2], 'szxy')
        currPose.pose.orientation.x = compensated_quat[0]
        currPose.pose.orientation.y = compensated_quat[1]
        currPose.pose.orientation.z = compensated_quat[2]
        currPose.pose.orientation.w = compensated_quat[3]
        
        # HOP UP: Lift 5mm to disengage from surface
        currPose.pose.position.z = currPose.pose.position.z + 5e-3
        rtde_help.goToPose_2Dhaptic(currPose)
        
        # HOP DOWN: Return to contact
        currPose.pose.position.z = currPose.pose.position.z - 5e-3 - pose_diff
        rtde_help.goToPose_2Dhaptic(currPose)
        rospy.sleep(args.hop_sleep)
        
        # ===== UPDATE STATE =====
        measuredCurrPose = rtde_help.getCurrentPose()
        T_curr = search_help.get_Tmat_from_Pose(measuredCurrPose)
        args.final_yaw = convert_yawAngle(search_help.get_yawRotation_from_T(T_curr))
        pose_diff = measuredCurrPose.pose.position.z - currPose.pose.position.z
        
        # Update orientation error
        q_measured = measuredCurrPose.pose.orientation
        q_commanded = currPose.pose.orientation
        measured_euler = tf.transformations.euler_from_quaternion([q_measured.x, q_measured.y, q_measured.z, q_measured.w], 'szxy')
        commanded_euler = tf.transformations.euler_from_quaternion([q_commanded.x, q_commanded.y, q_commanded.z, q_commanded.w], 'szxy')
        orientation_error = [measured_euler[i] - commanded_euler[i] for i in range(3)]
        
        # ===== UPDATE PATH LENGTH =====
        curr_position = measuredCurrPose.pose.position
        if prev_position is not None:
            dx = curr_position.x - prev_position.x
            dy = curr_position.y - prev_position.y
            path_length_2d += np.sqrt(dx**2 + dy**2)
            
            dz = curr_position.z - prev_position.z
            path_length_3d += np.sqrt(dx**2 + dy**2 + dz**2)
        
        prev_position = curr_position
        iteration_count += 1

        # ===== PROGRESS MONITORING =====
        if iteration_count % 10 == 0:
            if experiment_num:
                print(f"Experiment {experiment_num} - Iteration {iteration_count}/{max_steps}")
            else:
                print(f"Iteration {iteration_count}/{max_steps}")

    # ==================== CLEANUP AND FINALIZATION ====================
    rospy.sleep(0.2)
    dataLoggerEnable(False)
    rospy.sleep(0.2)
    P_help.stopSampling()
    targetPWM_Pub.publish(DUTYCYCLE_0)

    # Store results in args for data saving
    args.suction = suctionSuccessFlag
    args.iteration_count = iteration_count
    args.path_length_2d = path_length_2d
    args.path_length_3d = path_length_3d
    
    # Save .mat file with all logged data
    try:
        # Create a unique identifier for this trial
        trial_id = f"trial_{experiment_num}" if experiment_num else "trial"
        file_help.saveDataParams(args, appendTxt=trial_id)
        print(f"Saved .mat file for {trial_id}")
    except Exception as e:
        print(f"Warning: Failed to save .mat file: {e}")
    
    # Check if boundary was exceeded (if polygon_center_pose was provided)
    boundary_exceeded = False
    if polygon_center_pose is not None and not suctionSuccessFlag:
        # Check final position against boundary
        final_pose = rtde_help.getCurrentPose()
        current_x = final_pose.pose.position.x
        current_y = final_pose.pose.position.y
        center_x = polygon_center_pose.pose.position.x
        center_y = polygon_center_pose.pose.position.y
        
        dx = current_x - center_x
        dy = current_y - center_y
        displacement = np.sqrt(dx**2 + dy**2)
        boundary_limit = 25e-3  # 25mm boundary limit from polygon center
        boundary_exceeded = displacement > boundary_limit
    
    return {
        'success': suctionSuccessFlag,
        'iterations': iteration_count,
        'elapsed_time': args.elapsedTime if hasattr(args, 'elapsedTime') else 0,
        'final_yaw': args.final_yaw if hasattr(args, 'final_yaw') else 0,
        'path_length_2d': path_length_2d,
        'path_length_3d': path_length_3d,
        'boundary_exceeded': boundary_exceeded
    }


def get_polygon_position(polygon_row, polygon_col, base_disengage_position):
    """
    Calculate the position of a specific polygon in the 3x4 grid.
    
    Args:
        polygon_row: Row index (0, 1, 2)
        polygon_col: Column index (0, 1, 2, 3)
        base_disengage_position: Base position for polygon (0,0)
    
    Returns:
        List of [x, y, z] coordinates for the specified polygon
    """
    # Each polygon is 50mm apart
    polygon_spacing = 50e-3  # 50mm in meters
    
    # Calculate offset from base position
    x_offset = polygon_col * polygon_spacing
    y_offset = polygon_row * polygon_spacing
    
    # Apply offset to base position
    polygon_position = [
        base_disengage_position[0] + x_offset,
        base_disengage_position[1] + y_offset,
        base_disengage_position[2]  # Z remains the same
    ]
    
    return polygon_position


def polygon_number_to_coords(polygon_num):
    """
    Convert polygon number (1-12) to row/col coordinates.
    
    Args:
        polygon_num: Polygon number (1-12)
        
    Returns:
        Tuple of (row, col) coordinates
    """
    if polygon_num < 1 or polygon_num > 12:
        raise ValueError("Polygon number must be between 1 and 12")
    
    # Convert 1-based to 0-based indexing
    polygon_num -= 1
    
    # Calculate row and column
    row = polygon_num // 4  # 0, 1, 2
    col = polygon_num % 4   # 0, 1, 2, 3
    
    return row, col


def generate_random_initial_pose(disengage_pose, min_distance=1.0e-2, max_distance=1.9e-2, rtde_help=None, max_attempts=10):
    """
    Generate a random initial pose within the specified distance range from disengagePose.
    Validates poses to ensure they are reachable and within joint/pose limits.
    
    Args:
        disengage_pose: Base disengage pose
        min_distance: Minimum distance from disengagePose (default: 1.0cm)
        max_distance: Maximum distance from disengagePose (default: 2.0cm)
        rtde_help: RTDE helper for pose validation (optional)
        max_attempts: Maximum attempts to generate valid pose (default: 10)
    
    Returns:
        Tuple of (PoseStamped object, random_yaw)
    """
    for attempt in range(max_attempts):
        # Generate random distance within range
        distance = np.random.uniform(min_distance, max_distance)
        
        # Generate random angle (0 to 2π)
        angle = np.random.uniform(0, 2 * np.pi)
        
        # Calculate random offset
        x_offset = distance * np.cos(angle)
        y_offset = distance * np.sin(angle)
        
        # Generate random yaw for orientation (-150° to +150° to avoid extreme joint angles)
        random_yaw = np.random.uniform(-150*np.pi/180, 150*np.pi/180)
        
        # Create new pose
        random_pose = copy.deepcopy(disengage_pose)
        random_pose.pose.position.x += x_offset
        random_pose.pose.position.y += y_offset
        
        # Set random orientation (keep roll and pitch the same, only vary yaw)
        q = disengage_pose.pose.orientation
        current_euler = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w], 'szxy')
        new_yaw = current_euler[0] - random_yaw  # Subtract to add random variation to base orientation
        setOrientation = tf.transformations.quaternion_from_euler(new_yaw, current_euler[1], current_euler[2], 'szxy')
        random_pose.pose.orientation.x = setOrientation[0]
        random_pose.pose.orientation.y = setOrientation[1]
        random_pose.pose.orientation.z = setOrientation[2]
        random_pose.pose.orientation.w = setOrientation[3]
        
        # Validate pose if rtde_help is provided
        if rtde_help is not None:
            if rtde_help.validatePose(random_pose):
                print(f"      Generated valid pose after {attempt + 1} attempt(s)")
                return random_pose, random_yaw
            else:
                if attempt < max_attempts - 1:
                    print(f"      Generated pose {attempt + 1} failed validation (joint/pose limits), trying again...")
                    continue
        else:
            # No validation, return first generated pose
            return random_pose, random_yaw
    
    # If all attempts failed and validation was enabled, return the last generated pose with warning
    if rtde_help is not None:
        print(f"      Warning: Could not generate valid pose after {max_attempts} attempts, using last generated pose")
    return random_pose, random_yaw


def check_polygon_boundary(current_pose, polygon_disengage_pose, boundary_limit=2.5e-2):
    """
    Check if the current pose has moved beyond the boundary limit from the polygon disengage pose.
    
    Args:
        current_pose: Current robot pose
        polygon_disengage_pose: Initial disengage pose for this polygon
        boundary_limit: Maximum allowed distance in meters (default: 3cm)
    
    Returns:
        Tuple of (boundary_exceeded, displacement_distance, boundary_type)
    """
    # Calculate displacement from initial polygon position
    dx = current_pose.pose.position.x - polygon_disengage_pose.pose.position.x
    dy = current_pose.pose.position.y - polygon_disengage_pose.pose.position.y
    displacement = np.sqrt(dx**2 + dy**2)
    
    boundary_exceeded = displacement > boundary_limit
    
    if boundary_exceeded:
        boundary_type = "2D circular"
    else:
        boundary_type = "within_boundary"
    
    return boundary_exceeded, displacement, boundary_type


def validate_trial_start(rtde_help, P_help, search_help, targetPWM_Pub, dataLoggerEnable, 
                        initial_pose):
    """
    Validate that the initial position doesn't immediately achieve suction and has sufficient pressure gradient.
    This function validates a single pose - the outer loop handles generating new poses.
    
    Args:
        rtde_help: RTDE helper for robot control
        P_help: Helper for pressure sensor data
        search_help: Helper for 2D haptic search computations
        targetPWM_Pub: ROS publisher for vacuum pump PWM control
        dataLoggerEnable: ROS service proxy for enabling/disabling data logging
        initial_pose: Initial pose to validate
    
    Returns:
        Tuple of (validated_pose, validation_success)
    """
    P_vac = search_help.P_vac
    DUTYCYCLE_100 = 100
    DUTYCYCLE_0 = 0
    PRESSURE_GRADIENT_THRESHOLD = 20  # Pa - minimum pressure difference between chambers
    
    # Move to the initial pose
    rtde_help.goToPose_2Dhaptic(initial_pose)
    rospy.sleep(0.5)  # Allow robot to settle
    
    # Start vacuum and data logging
    targetPWM_Pub.publish(DUTYCYCLE_100)
    P_help.startSampling()
    rospy.sleep(0.5)
    P_help.setNowAsOffset()
    dataLoggerEnable(True)
    
    # Move down to engage position (5mm below disengage) to check pressure properly
    engage_position = copy.deepcopy(initial_pose.pose.position)
    engage_position.z = engage_position.z - 5e-3
    engage_pose = rtde_help.getPoseObj(engage_position, initial_pose.pose.orientation)
    rtde_help.goToPose_2Dhaptic(engage_pose)
    rospy.sleep(0.5)  # Allow pressure to stabilize
    
    # Check pressure conditions at engage position
    P_array = P_help.four_pressure
    
    # Apply same pressure processing as controllers for consistent null gradient detection
    P_processed = process_pressure_data(P_array, search_help)
    
    reached_vacuum = all(np.array(P_array) < P_vac)
    
    # Check for null gradient using processed pressure data (same as controllers)
    pressure_range = max(P_processed) - min(P_processed)
    has_null_gradient = pressure_range < PRESSURE_GRADIENT_THRESHOLD
    
    # Move back to disengage position
    rtde_help.goToPose_2Dhaptic(initial_pose)
    rospy.sleep(0.2)
    
    # Stop vacuum and data logging
    dataLoggerEnable(False)
    P_help.stopSampling()
    targetPWM_Pub.publish(DUTYCYCLE_0)
    
    if not reached_vacuum and not has_null_gradient:
        # Valid starting position - no immediate suction and sufficient pressure gradient
        print(f"  Validation successful")
        return initial_pose, True
    else:
        # Invalid starting position - either immediate suction or null gradient
        if reached_vacuum:
            print(f"  Validation failed: Immediate suction detected")
            #turn off vacuum
            targetPWM_Pub.publish(DUTYCYCLE_0)
        elif has_null_gradient:
            print(f"  Validation failed: Null gradient detected (range: {pressure_range:.1f} Pa < {PRESSURE_GRADIENT_THRESHOLD} Pa)")
            #turn off vacuum
            targetPWM_Pub.publish(DUTYCYCLE_0)
        return None, False


def test_polygon_position(args, rtde_help, polygon_row, polygon_col):
    """
    Test the position of a polygon by moving the robot arm to it.
    This mode is for visual testing of polygon locations - no vacuum, no data saved.
    
    Args:
        args: Command line arguments
        rtde_help: RTDE helper for robot control
        polygon_row: Row index of the polygon (0, 1, 2)
        polygon_col: Column index of the polygon (0, 1, 2, 3)
    
    Returns:
        None (visual test only, no data saved)
    """
    print(f"\n--- Position Test for Polygon ({polygon_row},{polygon_col}) ---")
    
    # Get base disengage position for polygons
    base_disengage_position = get_disEngagePosition('polygons')
    
    # Calculate polygon position
    polygon_position = get_polygon_position(polygon_row, polygon_col, base_disengage_position)
    
    # Set default yaw angle based on channel
    # if args.ch == 3:
    #     default_yaw = pi/2 - 60*pi/180
    # elif args.ch == 4:
    #     default_yaw = pi/2 - 45*pi/180
    # elif args.ch == 5:
    #     default_yaw = pi/2 - 90*pi/180
    # elif args.ch == 6:
    #     default_yaw = pi/2 - 60*pi/180
    # polygon objects do not need to be oriented, since it will use random orientation for each trial
    default_yaw = pi/2 + 45*pi/180
    
    setOrientation = tf.transformations.quaternion_from_euler(default_yaw, pi, 0, 'szxy')
    polygon_disengage_pose = rtde_help.getPoseObj(polygon_position, setOrientation)
    
    # Move to polygon disengage position
    print(f"Moving to polygon ({polygon_row},{polygon_col}) position...")
    print(f"  Position: X={polygon_position[0]*1000:.1f}mm, Y={polygon_position[1]*1000:.1f}mm, Z={polygon_position[2]*1000:.1f}mm")
    rtde_help.goToPose_2Dhaptic(polygon_disengage_pose)
    rospy.sleep(1.0)
    
    # Move down to engage position (5mm below disengage) for visual inspection
    engage_position = copy.deepcopy(polygon_disengage_pose.pose.position)
    engage_position.z = engage_position.z - 5e-3
    engage_pose = rtde_help.getPoseObj(engage_position, polygon_disengage_pose.pose.orientation)
    rtde_help.goToPose_2Dhaptic(engage_pose)
    
    # Hold position for 2 seconds for visual inspection
    print(f"  Holding position for 2 seconds for visual inspection...")
    rospy.sleep(2.0)
    
    # Move back to disengage position
    rtde_help.goToPose_2Dhaptic(polygon_disengage_pose)
    rospy.sleep(0.5)
    
    # Display completion message
    print(f"  Position test completed for polygon ({polygon_row},{polygon_col})")
    
    return None


def run_single_polygon_experiment(args, rtde_help, search_help, P_help, targetPWM_Pub,
                                 dataLoggerEnable, file_help, rl_controllers_dict, 
                                 polygon_row, polygon_col, controller_name, 
                                 num_trials=None):
    """
    Run experiments on a single polygon with multiple trials.
    
    Args:
        args: Command line arguments
        rtde_help: RTDE helper for robot control
        search_help: Helper for 2D haptic search computations
        P_help: Helper for pressure sensor data
        targetPWM_Pub: ROS publisher for vacuum pump PWM control
        dataLoggerEnable: ROS service proxy for enabling/disabling data logging
        file_help: Helper for saving experiment data
        rl_controllers_dict: Dictionary of RL controllers {controller_name: rl_controller}
        polygon_row: Row index of the polygon (0, 1, 2)
        polygon_col: Column index of the polygon (0, 1, 2, 3)
        controller_name: Name of the controller being tested
        num_trials: Number of trials to run (if None, uses debug_mode to determine)
    
    Returns:
        Dictionary containing results for this polygon
    """
    # Determine number of trials based on debug mode
    if num_trials is None:
        num_trials = 1 if args.debug_mode else 10
    
    print(f"\n--- Polygon ({polygon_row},{polygon_col}) - {controller_name} controller ---")
    if args.debug_mode:
        print(f"  DEBUG MODE: Running only {num_trials} trial per polygon (full haptic search)")
    
    # Get base disengage position for polygons
    base_disengage_position = get_disEngagePosition('polygons')
    
    # Calculate polygon position
    polygon_position = get_polygon_position(polygon_row, polygon_col, base_disengage_position)
    
    # Set default yaw angle based on channel
    # if args.ch == 3:
    #     default_yaw = pi/2 - 60*pi/180
    # elif args.ch == 4:
    #     default_yaw = pi/2 - 45*pi/180
    # elif args.ch == 5:
    #     default_yaw = pi/2 - 90*pi/180
    # elif args.ch == 6:
    #     default_yaw = pi/2 - 60*pi/180
    # polygon objects do not need to be oriented, since it will use random orientation for each trial
    default_yaw = pi/2 + 45*pi/180
    
    setOrientation = tf.transformations.quaternion_from_euler(default_yaw, pi, 0, 'szxy')
    polygon_disengage_pose = rtde_help.getPoseObj(polygon_position, setOrientation)
    
    # Move to polygon disengage position
    print(f"Moving to polygon ({polygon_row},{polygon_col}) position...")
    rtde_help.goToPose_2Dhaptic(polygon_disengage_pose)
    rospy.sleep(1.0)
    
    # Store results for this polygon
    polygon_results = {
        'polygon_row': polygon_row,
        'polygon_col': polygon_col,
        'controller': controller_name,
        'chamber_count': args.ch,
        'random_seed': args.seed,
        'trials': [],
        'successful_trials': 0,
        'failed_trials': 0,
        'total_trials': 0,
        'success_rate': 0.0,
        'avg_iterations': 0.0,
        'avg_path_length_2d': 0.0,
        'avg_path_length_3d': 0.0,
        'avg_elapsed_time': 0.0
    }
    
    # Variable to store the last trial's random pose (for transition after last polygon)
    last_trial_random_pose = None
    last_trial_random_yaw = None
    
    # Run trials
    for trial_num in range(1, num_trials + 1):
        print(f"  Trial {trial_num}/{num_trials}")
        
        # Validation loop - keep trying until we get a valid starting position
        validation_successful = False
        validation_attempts = 0
        max_validation_attempts = 10
        
        while not validation_successful and validation_attempts < max_validation_attempts:
            validation_attempts += 1
            print(f"    Validation attempt {validation_attempts}/{max_validation_attempts} for trial {trial_num}...")
            
            # Check if robot is in extreme orientation and reset if needed
            if validation_attempts == 3:  # After 2 failed attempts, check orientation
                current_pose = rtde_help.getCurrentPose()
                current_euler = tf.transformations.euler_from_quaternion([
                    current_pose.pose.orientation.x, current_pose.pose.orientation.y, 
                    current_pose.pose.orientation.z, current_pose.pose.orientation.w
                ], 'szxy')
                current_yaw_deg = abs(current_euler[0] * 180 / np.pi)
                
                if current_yaw_deg > 90:  # If yaw is more than 90 degrees
                    print(f"      Robot in extreme orientation (yaw={current_yaw_deg:.1f}°), resetting...")
                    polygon_position = [polygon_disengage_pose.pose.position.x, 
                                      polygon_disengage_pose.pose.position.y, 
                                      polygon_disengage_pose.pose.position.z]
                    reset_robot_to_safe_orientation(rtde_help, polygon_position)
            
            # Always start from polygon's initial pose for each new attempt
            print(f"      Moving to polygon initial pose...")
            rtde_help.goToPose_2Dhaptic(polygon_disengage_pose)
            rospy.sleep(0.5)  # Allow robot to settle at polygon initial pose
            
            # Generate NEW random pose for this validation attempt (with joint/pose limit validation)
            print(f"      Generating new random pose...")
            random_pose, random_yaw = generate_random_initial_pose(polygon_disengage_pose, rtde_help=rtde_help)
            
            # Store initial yaw for reverse rotation calculation
            initial_yaw = convert_yawAngle(search_help.get_yawRotation_from_T(search_help.get_Tmat_from_Pose(random_pose)))
            
            # Move to higher position before going to random pose
            current_pose = rtde_help.getCurrentPose()
            high_pose = copy.deepcopy(current_pose)
            high_pose.pose.position.z = current_pose.pose.position.z + 5e-3  # 5mm higher
            rtde_help.goToPose_2Dhaptic(high_pose)
            rospy.sleep(0.3)  # Brief pause at higher position
            
            # Move to random pose (elevated)
            print(f"      Moving to new random pose...")
            random_pose_elevated = copy.deepcopy(random_pose)
            random_pose_elevated.pose.position.z = high_pose.pose.position.z  # Keep elevated
            rtde_help.goToPose_2Dhaptic(random_pose_elevated)
            rospy.sleep(0.5)
            
            # Lower to random pose and validate trial start (ensure no immediate suction)
            validated_pose, validation_success = validate_trial_start(
                rtde_help, P_help, search_help, targetPWM_Pub, dataLoggerEnable, random_pose
            )
            
            if validation_success:
                # Validation successful
                validation_successful = True
                print(f"      Validation successful after {validation_attempts} attempts")
            else:
                # Validation failed - will try new random pose in next iteration
                print(f"      Validation failed, will try new random pose in next attempt...")
        
        if not validation_successful:
            print(f"    Warning: Could not find valid starting position after {max_validation_attempts} attempts")
            print(f"    Attempting to reset robot to safe orientation...")
            
            # Reset robot to safe orientation
            polygon_position = [polygon_disengage_pose.pose.position.x, 
                              polygon_disengage_pose.pose.position.y, 
                              polygon_disengage_pose.pose.position.z]
            
            if reset_robot_to_safe_orientation(rtde_help, polygon_position):
                print(f"    Robot reset successful, trying one more validation attempt...")
                # Try one more time with reset robot
                random_pose, random_yaw = generate_random_initial_pose(polygon_disengage_pose, rtde_help=rtde_help)
                validated_pose, validation_success = validate_trial_start(
                    rtde_help, P_help, search_help, targetPWM_Pub, dataLoggerEnable, random_pose
                )
                if validation_success:
                    print(f"    Validation successful after robot reset!")
                    validation_successful = True
                else:
                    print(f"    Validation still failed after reset, using last generated pose")
                    validated_pose = random_pose
            else:
                print(f"    Robot reset failed, using last generated pose")
                validated_pose = random_pose
        
        # Update args for this trial
        args.controller = controller_name
        args.random_yaw = random_yaw
        
        # Run haptic search experiment
        result = run_haptic_search_loop(
            args, rtde_help, search_help, P_help, targetPWM_Pub,
            dataLoggerEnable, file_help, validated_pose, rl_controllers_dict,
            experiment_num=trial_num, random_yaw=random_yaw, polygon_center_pose=polygon_disengage_pose
        )
        
        # Check if trial was abandoned due to joint limit violation
        if result.get('joint_limit_violation', False):
            print(f"    Trial {trial_num} abandoned due to joint limit violation, generating new random pose...")
            # Generate a new random pose for this trial
            new_random_pose, new_random_yaw = generate_random_initial_pose(polygon_disengage_pose, rtde_help=rtde_help)
            
            # Validate the new pose
            new_validated_pose, new_validation_success = validate_trial_start(
                rtde_help, P_help, search_help, targetPWM_Pub, dataLoggerEnable, new_random_pose
            )
            
            if new_validation_success:
                print(f"    New random pose validated, retrying trial {trial_num}...")
                # Update args for the new trial
                args.random_yaw = new_random_yaw
                
                # Run haptic search experiment with new pose
                result = run_haptic_search_loop(
                    args, rtde_help, search_help, P_help, targetPWM_Pub,
                    dataLoggerEnable, file_help, new_validated_pose, rl_controllers_dict,
                    experiment_num=trial_num, random_yaw=new_random_yaw, polygon_center_pose=polygon_disengage_pose
                )
            else:
                print(f"    Warning: New random pose also failed validation, using original result")
        
        # Return to polygon default position after each trial
        # For trials 1-9: return prepares for next trial
        # For trial 10: return prepares for transition to next polygon
        # Step 1: Move to higher position first
        current_pose = rtde_help.getCurrentPose()
        high_pose = copy.deepcopy(current_pose)
        high_pose.pose.position.z = current_pose.pose.position.z + 5e-3  # 5mm higher
        rtde_help.goToPose_2Dhaptic(high_pose)
        rospy.sleep(0.3)
        
        # Step 2: Move to random pose (with current orientation from haptic search) - elevated
        random_pose_with_current_orientation = copy.deepcopy(random_pose)
        # Keep the current orientation from haptic search
        current_orientation = current_pose.pose.orientation
        random_pose_with_current_orientation.pose.orientation = current_orientation
        # Keep elevated position (5mm above contact)
        random_pose_with_current_orientation.pose.position.z = high_pose.pose.position.z
        rtde_help.goToPose_2Dhaptic(random_pose_with_current_orientation)
        rospy.sleep(0.5)
        
        # Step 3: Move to default pose (with original random orientation) - elevated
        default_pose_elevated = copy.deepcopy(polygon_disengage_pose)
        default_pose_elevated.pose.position.z = high_pose.pose.position.z  # Keep elevated
        rtde_help.goToPose_2Dhaptic(default_pose_elevated)
        rospy.sleep(0.5)
        
        # Step 4: Lower to default position
        # For trials 1-9: prepares for next trial
        # For trial 10: prepares for transition to next polygon
        rtde_help.goToPose_2Dhaptic(polygon_disengage_pose)
        rospy.sleep(0.5)  # Allow robot to settle at default position
        
        # Boundary checking is now handled during haptic search
        
        # Store trial result
        trial_result = {
            'trial_num': trial_num,
            'chamber_count': args.ch,
            'random_seed': args.seed,
            'success': result['success'],
            'iterations': result['iterations'],
            'elapsed_time': result['elapsed_time'],
            'path_length_2d': result['path_length_2d'],
            'path_length_3d': result['path_length_3d'],
            'final_yaw': result['final_yaw'],
            'random_yaw': random_yaw,
            'validation_attempts': validation_attempts,
            'boundary_exceeded': result.get('boundary_exceeded', False)
        }
        
        polygon_results['trials'].append(trial_result)
        polygon_results['total_trials'] += 1
        
        # Save random pose for this trial (for potential use in transition after last polygon)
        last_trial_random_pose = random_pose
        last_trial_random_yaw = random_yaw
        
        if result['success']:
            polygon_results['successful_trials'] += 1
        else:
            polygon_results['failed_trials'] += 1
        
        # Calculate and print current success rate
        current_success_rate = polygon_results['successful_trials'] / polygon_results['total_trials'] * 100
        print(f"    Trial {trial_num} result: {'SUCCESS' if result['success'] else 'FAILED'} - Current polygon success rate: {current_success_rate:.1f}% ({polygon_results['successful_trials']}/{polygon_results['total_trials']})")
        
        # Save individual trial data
        trial_filename = f"polygon_{polygon_row}_{polygon_col}_trial_{trial_num:02d}_{controller_name}_ch{args.ch}.json"
        trial_path = os.path.join(file_help.ResultSavingDirectory, trial_filename)
        
        with open(trial_path, 'w') as f:
            import json
            json.dump(trial_result, f, indent=2, default=str)
    
    # Calculate summary statistics for this polygon
    successful_trials = [t for t in polygon_results['trials'] if t['success']]
    boundary_failures = [t for t in polygon_results['trials'] if t['boundary_exceeded']]
    
    polygon_results['success_rate'] = polygon_results['successful_trials'] / polygon_results['total_trials'] * 100
    polygon_results['boundary_failures'] = len(boundary_failures)
    polygon_results['boundary_failure_rate'] = len(boundary_failures) / polygon_results['total_trials'] * 100
    
    if successful_trials:
        polygon_results['avg_iterations'] = np.mean([t['iterations'] for t in successful_trials])
        polygon_results['avg_path_length_2d'] = np.mean([t['path_length_2d'] for t in successful_trials])
        polygon_results['avg_path_length_3d'] = np.mean([t['path_length_3d'] for t in successful_trials])
        polygon_results['avg_elapsed_time'] = np.mean([t['elapsed_time'] for t in successful_trials])
    
    # Save polygon summary
    polygon_summary_filename = f"polygon_{polygon_row}_{polygon_col}_summary_{controller_name}_ch{args.ch}.json"
    polygon_summary_path = os.path.join(file_help.ResultSavingDirectory, polygon_summary_filename)
    
    with open(polygon_summary_path, 'w') as f:
        import json
        json.dump(polygon_results, f, indent=2, default=str)
    
    print(f"  Polygon ({polygon_row},{polygon_col}) completed: {polygon_results['successful_trials']}/{polygon_results['total_trials']} successful ({polygon_results['success_rate']:.1f}%), {polygon_results['boundary_failures']} boundary failures ({polygon_results['boundary_failure_rate']:.1f}%)")
    
    # Return polygon results along with last trial's random pose for transition
    return polygon_results, last_trial_random_pose, last_trial_random_yaw


def run_polygon_visual_test(args, rtde_help):
    """
    Run visual tests for all polygons in the grid.
    
    This function tests the position of each polygon by moving the robot arm to it.
    This is for visual inspection only - no vacuum, no data saved.
    Robot moves 5mm higher when transitioning between polygons for clearance.
    
    Press Ctrl+C at any time to stop the test.
    
    Args:
        args: Command line arguments
        rtde_help: RTDE helper for robot control
    
    Returns:
        None (visual test only, no data saved)
    """
    print(f"Starting polygon position test")
    print(f"Polygons: 3x4 grid (12 total)")
    print(f"Chambers per polygon: {args.ch}")
    print("This is a visual test only - no vacuum, no data will be saved")
    print("Robot will move 5mm higher between polygons for clearance")
    print("Press Ctrl+C at any time to stop the test")
    print("=" * 60)
    
    try:
        # Test each polygon
        for row in range(3):
            for col in range(4):
                polygon_key = f"polygon_{row}_{col}"
                print(f"\n--- Testing {polygon_key} ---")
                
                # Run position test on this polygon
                test_polygon_position(args, rtde_help, row, col)
                
                # Move to higher position before going to next polygon (except for the last one)
                if not (row == 2 and col == 3):  # Not the last polygon
                    print(f"  Moving to higher position for clearance...")
                    # Get current position and move 5mm higher
                    current_pose = rtde_help.getCurrentPose()
                    high_pose = copy.deepcopy(current_pose)
                    high_pose.pose.position.z = current_pose.pose.position.z + 5e-3  # 5mm higher
                    rtde_help.goToPose_2Dhaptic(high_pose)
                    rospy.sleep(0.5)  # Brief pause at higher position
        
        # Return to polygon (0,0) after completing all tests
        print(f"\n--- Returning to polygon (0,0) ---")
        
        # Move to higher position first (10mm higher than current position)
        print(f"  Moving to higher position (10mm) for safe return...")
        current_pose = rtde_help.getCurrentPose()
        high_pose = copy.deepcopy(current_pose)
        high_pose.pose.position.z = current_pose.pose.position.z + 10e-3  # 10mm higher
        rtde_help.goToPose_2Dhaptic(high_pose)
        rospy.sleep(0.5)
        
        # Move to polygon (0,0) position
        print(f"  Moving to polygon (0,0) position...")
        base_disengage_position = get_disEngagePosition('polygons')
        polygon_position = get_polygon_position(0, 0, base_disengage_position)
        
        # Set default yaw angle based on channel
        # if args.ch == 3:
        #     default_yaw = pi/2 - 60*pi/180
        # elif args.ch == 4:
        #     default_yaw = pi/2 - 45*pi/180
        # elif args.ch == 5:
        #     default_yaw = pi/2 - 90*pi/180
        # elif args.ch == 6:
        #     default_yaw = pi/2 - 60*pi/180
        # polygon objects do not need to be oriented, since it will use random orientation for each trial
        default_yaw = pi/2 + 45*pi/180
        
        setOrientation = tf.transformations.quaternion_from_euler(default_yaw, pi, 0, 'szxy')
        return_pose = rtde_help.getPoseObj(polygon_position, setOrientation)
        rtde_help.goToPose_2Dhaptic(return_pose)
        rospy.sleep(1.0)
        
        print(f"  Returned to polygon (0,0) position")
        print(f"  Position: X={polygon_position[0]*1000:.1f}mm, Y={polygon_position[1]*1000:.1f}mm, Z={polygon_position[2]*1000:.1f}mm")
        
        print(f"\n{'='*60}")
        print("POLYGON POSITION TEST COMPLETED!")
        print(f"{'='*60}")
        print("All 12 polygons have been tested visually.")
        print("Robot has returned to polygon (0,0) position.")
        print("No vacuum was used - this was a position inspection only.")
        
    except KeyboardInterrupt:
        print(f"\n\n{'='*60}")
        print("POLYGON POSITION TEST INTERRUPTED BY USER!")
        print(f"{'='*60}")
        print("Test stopped by Ctrl+C. Robot will remain at current position.")
        print("You can restart the test anytime.")
        
    return None


def run_polygon_grid_experiment(args, rtde_help, search_help, P_help, targetPWM_Pub,
                               dataLoggerEnable, file_help, rl_controller, 
                               controllers=['greedy', 'rl_hyaw_momentum']):
    """
    Run complete polygon grid experiment with multiple controllers.
    
    Args:
        args: Command line arguments
        rtde_help: RTDE helper for robot control
        search_help: Helper for 2D haptic search computations
        P_help: Helper for pressure sensor data
        targetPWM_Pub: ROS publisher for vacuum pump PWM control
        dataLoggerEnable: ROS service proxy for enabling/disabling data logging
        file_help: Helper for saving experiment data
        rl_controller: RL controller object
        controllers: List of controllers to test (default: ['greedy', 'rl_hyaw_momentum'])
    
    Returns:
        Dictionary containing complete experiment results
    """
    print(f"Starting polygon grid experiment")
    print(f"Controllers to test: {', '.join(controllers)}")
    print(f"Polygons: 3x4 grid (12 total)")
    trials_per_polygon = 2 if args.debug_mode else 10
    print(f"Trials per polygon: {trials_per_polygon}")
    if args.debug_mode:
        print(f"DEBUG MODE ENABLED: Fast testing with 1 trial per polygon (full haptic search still performed)")
    print("=" * 60)
    
    # Initialize RL controllers if needed
    rl_controllers = {}
    controllers_to_remove = []
    
    for controller_name in controllers:
        if controller_name.startswith('rl_'):
            try:
                # Extract model type from RL controller name
                model_type = controller_name.replace('rl_', '')
                rl_controller = create_rl_controller(args.ch, model_type)
                rl_controllers[controller_name] = rl_controller
                print(f"RL controller initialized: ch{args.ch}_{model_type}")
            except Exception as e:
                print(f"Failed to initialize RL controller {controller_name}: {e}")
                print(f"Removing {controller_name} from test list")
                controllers_to_remove.append(controller_name)
    
    # Remove failed RL controllers from the list
    for controller_name in controllers_to_remove:
        controllers.remove(controller_name)
    
    # Create main experiment folder (single directory for all data)
    from datetime import datetime
    date_folder = datetime.now().strftime("%y%m%d")
    time_folder = datetime.now().strftime("%H%M%S")
    base_folder = os.path.expanduser('~') + '/SuctionExperiment'
    seed_suffix = f"_seed{args.seed}" if args.seed is not None else "_seedNone"
    main_experiment_folder = os.path.join(base_folder, date_folder, f"polygon_grid_experiment_ch{args.ch}{seed_suffix}_{time_folder}")
    
    if not os.path.exists(main_experiment_folder):
        os.makedirs(main_experiment_folder)
    
    print(f"All data will be saved to: {main_experiment_folder}")
    
    # Store all results
    all_results = {
        'experiment_info': {
            'start_time': datetime.now().isoformat(),
            'controllers': controllers,
            'channels': args.ch,
            'trials_per_polygon': trials_per_polygon,
            'total_polygons': 12,
            'debug_mode': args.debug_mode,
            'data_directory': main_experiment_folder,
            'random_seed': args.seed
        },
        'controller_results': {}
    }
    
    # Test each controller
    for controller_idx, controller_name in enumerate(controllers, 1):
        print(f"\n{'='*20} CONTROLLER {controller_idx}/{len(controllers)}: {controller_name.upper()} {'='*20}")
        
        # Create controller-specific subdirectory
        controller_subfolder = os.path.join(main_experiment_folder, f"controller_{controller_name}_ch{args.ch}")
        if not os.path.exists(controller_subfolder):
            os.makedirs(controller_subfolder)
        
        controller_file_help = fileSaveHelp(savingFolderName="")
        controller_file_help.ResultSavingDirectory = controller_subfolder
        
        print(f"Controller data will be saved to: {controller_subfolder}")
        
        # Store results for this controller
        controller_results = {
            'controller': controller_name,
            'chamber_count': args.ch,
            'random_seed': args.seed,
            'polygons': {},
            'overall_stats': {}
        }
        
        # Test each polygon (starting from specified polygon)
        start_row, start_col = polygon_number_to_coords(args.start_polygon)
        print(f"Starting from polygon {args.start_polygon} (row={start_row}, col={start_col})")
        
        for row in range(3):
            for col in range(4):
                # Skip polygons before the starting polygon
                polygon_num = row * 4 + col + 1  # Convert to 1-based polygon number
                if polygon_num < args.start_polygon:
                    print(f"Skipping polygon {polygon_num} (row={row}, col={col}) - before start polygon {args.start_polygon}")
                    continue
                
                polygon_key = f"polygon_{row}_{col}"
                print(f"\n--- Testing {polygon_key} (polygon {polygon_num}) ---")
                
                # Run experiments on this polygon
                polygon_result, last_random_pose, last_random_yaw = run_single_polygon_experiment(
                    args, rtde_help, search_help, P_help, targetPWM_Pub,
                    dataLoggerEnable, controller_file_help, rl_controllers,
                    row, col, controller_name, num_trials=trials_per_polygon
                )
                
                controller_results['polygons'][polygon_key] = polygon_result
                
                # Calculate and print running success rate
                all_trials_so_far = []
                for polygon_data in controller_results['polygons'].values():
                    all_trials_so_far.extend(polygon_data['trials'])
                
                if all_trials_so_far:
                    successful_so_far = len([t for t in all_trials_so_far if t['success']])
                    running_success_rate = successful_so_far / len(all_trials_so_far) * 100
                    print(f"  Running success rate: {running_success_rate:.1f}% ({successful_so_far}/{len(all_trials_so_far)})")
                
                # Move to higher position and turn off vacuum before next polygon
                # Turn off vacuum
                targetPWM_Pub.publish(0)
                rospy.sleep(0.2)

                if not (row == 2 and col == 3):  # Not the last polygon
                    # Get current position and move 10mm higher
                    current_pose = rtde_help.getCurrentPose()
                    high_pose = copy.deepcopy(current_pose)
                    high_pose.pose.position.z = current_pose.pose.position.z + 10e-3  # 10mm higher
                    rtde_help.goToPose_2Dhaptic(high_pose)
                    rospy.sleep(0.5)  # Brief pause at higher position
                else:
                    # Last polygon (2,3) - follow same pattern as between trials: random pose -> original pose -> next polygon
                    print(f"  Last polygon (2,3) completed - following transition pattern...")
                    current_pose = rtde_help.getCurrentPose()
                    high_pose = copy.deepcopy(current_pose)
                    high_pose.pose.position.z = current_pose.pose.position.z + 10e-3  # 10mm higher
                    rtde_help.goToPose_2Dhaptic(high_pose)
                    rospy.sleep(0.5)
                    
                    # Step 1: Go to last trial's random pose (elevated) - using the 10th trial's random pose
                    print(f"  Moving to last trial's random pose (elevated)...")
                    random_pose_elevated = copy.deepcopy(last_random_pose)
                    # Keep the current orientation from haptic search
                    current_orientation = current_pose.pose.orientation
                    random_pose_elevated.pose.orientation = current_orientation
                    random_pose_elevated.pose.position.z = high_pose.pose.position.z  # Keep elevated
                    rtde_help.goToPose_2Dhaptic(random_pose_elevated)
                    rospy.sleep(0.5)
                    
                    # Step 2: Return to polygon (2,3) original pose (elevated)
                    print(f"  Returning to polygon (2,3) original pose (elevated)...")
                    base_disengage_position = get_disEngagePosition('polygons')
                    polygon_12_position = get_polygon_position(2, 3, base_disengage_position)
                    setOrientation = tf.transformations.quaternion_from_euler(pi/2 + 45*pi/180, pi, 0, 'szxy')
                    polygon_12_disengage_pose = rtde_help.getPoseObj(polygon_12_position, setOrientation)
                    polygon_12_elevated = copy.deepcopy(polygon_12_disengage_pose)
                    polygon_12_elevated.pose.position.z = high_pose.pose.position.z  # Keep elevated
                    rtde_help.goToPose_2Dhaptic(polygon_12_elevated)
                    rospy.sleep(0.5)
                    
                    # Step 3: Move to polygon (0,0) elevated first
                    print(f"  Moving to polygon (0,0) (elevated) for next controller...")
                    polygon_0_position = get_polygon_position(0, 0, base_disengage_position)
                    default_yaw = pi/2 + 45*pi/180
                    setOrientation = tf.transformations.quaternion_from_euler(default_yaw, pi, 0, 'szxy')
                    return_pose = rtde_help.getPoseObj(polygon_0_position, setOrientation)
                    return_pose.pose.position.z = high_pose.pose.position.z  # Keep elevated
                    
                    print(f"  Position: X={polygon_0_position[0]*1000:.1f}mm, Y={polygon_0_position[1]*1000:.1f}mm, Z={polygon_0_position[2]*1000:.1f}mm")
                    rtde_help.goToPose_2Dhaptic(return_pose)
                    rospy.sleep(0.5)
                    
                    # Step 4: Lower to polygon (0,0) initial pose
                    print(f"  Lowering to polygon (0,0) initial pose...")
                    polygon_0_initial_pose = rtde_help.getPoseObj(polygon_0_position, setOrientation)
                    rtde_help.goToPose_2Dhaptic(polygon_0_initial_pose)
                    rospy.sleep(0.5)
                    
                    print(f"  Ready for next controller at polygon (0,0)")
                    
                    
        
        # Calculate overall statistics for this controller
        all_polygon_trials = []
        for polygon_data in controller_results['polygons'].values():
            all_polygon_trials.extend(polygon_data['trials'])
        
        successful_trials = [t for t in all_polygon_trials if t['success']]
        boundary_failures = [t for t in all_polygon_trials if t['boundary_exceeded']]
        total_trials = len(all_polygon_trials)
        
        controller_results['overall_stats'] = {
            'total_trials': total_trials,
            'successful_trials': len(successful_trials),
            'failed_trials': total_trials - len(successful_trials),
            'boundary_failures': len(boundary_failures),
            'overall_success_rate': len(successful_trials) / total_trials * 100 if total_trials > 0 else 0,
            'boundary_failure_rate': len(boundary_failures) / total_trials * 100 if total_trials > 0 else 0,
            'avg_iterations': np.mean([t['iterations'] for t in successful_trials]) if successful_trials else 0,
            'avg_path_length_2d': np.mean([t['path_length_2d'] for t in successful_trials]) if successful_trials else 0,
            'avg_path_length_3d': np.mean([t['path_length_3d'] for t in successful_trials]) if successful_trials else 0,
            'avg_elapsed_time': np.mean([t['elapsed_time'] for t in successful_trials]) if successful_trials else 0
        }
        
        # Save controller summary
        controller_summary_filename = f"controller_{controller_name}_summary_ch{args.ch}.json"
        controller_summary_path = os.path.join(controller_file_help.ResultSavingDirectory, controller_summary_filename)
        
        with open(controller_summary_path, 'w') as f:
            import json
            json.dump(controller_results, f, indent=2, default=str)
        
        all_results['controller_results'][controller_name] = controller_results
        
        print(f"\n{controller_name} controller completed!")
        print(f"Overall success rate: {controller_results['overall_stats']['overall_success_rate']:.1f}%")
        print(f"Boundary failures: {controller_results['overall_stats']['boundary_failures']} ({controller_results['overall_stats']['boundary_failure_rate']:.1f}%)")
        print(f"Average iterations: {controller_results['overall_stats']['avg_iterations']:.1f}")
        print(f"Average 2D path length: {controller_results['overall_stats']['avg_path_length_2d']*1000:.1f} mm")
        
        # Return to polygon (0,0) after completing all tests for this controller
        print(f"\n--- Returning to polygon (0,0) after {controller_name} controller ---")
        
        # Move to higher position first (10mm higher than current position)
        print(f"  Moving to higher position (10mm) for safe return...")
        current_pose = rtde_help.getCurrentPose()
        high_pose = copy.deepcopy(current_pose)
        high_pose.pose.position.z = current_pose.pose.position.z + 10e-3  # 10mm higher
        rtde_help.goToPose_2Dhaptic(high_pose)
        rospy.sleep(0.5)
        
        # Move to polygon (0,0) position
        polygon_position = get_polygon_position(0, 0, base_disengage_position)
        
        # Set default yaw angle for return
        default_yaw = pi/2 + 45*pi/180  # Same as experiment default
        setOrientation = tf.transformations.quaternion_from_euler(default_yaw, pi, 0, 'szxy')
        return_pose = rtde_help.getPoseObj(polygon_position, setOrientation)
        
        print(f"  Moving to polygon (0,0) position...")
        print(f"  Position: X={polygon_position[0]*1000:.1f}mm, Y={polygon_position[1]*1000:.1f}mm, Z={polygon_position[2]*1000:.1f}mm")
        rtde_help.goToPose_2Dhaptic(return_pose)
        rospy.sleep(1.0)
        
        # Verify and correct orientation if needed
        current_pose = rtde_help.getCurrentPose()
        current_euler = tf.transformations.euler_from_quaternion([
            current_pose.pose.orientation.x, current_pose.pose.orientation.y, 
            current_pose.pose.orientation.z, current_pose.pose.orientation.w
        ], 'szxy')
        current_yaw = current_euler[0] * 180 / np.pi
        
        target_yaw = default_yaw * 180 / np.pi
        yaw_error = abs(current_yaw - target_yaw)
        
        if yaw_error > 5.0:  # If yaw error is more than 5 degrees
            print(f"  Correcting orientation: current yaw={current_yaw:.1f}°, target={target_yaw:.1f}°")
            # Move to the exact target orientation
            rtde_help.goToPose_2Dhaptic(return_pose)
            rospy.sleep(0.5)
        
        print(f"  Returned to polygon (0,0) position with correct orientation")
        
        # Reset yaw tracking for next controller to prevent cumulative yaw issues
        search_help.reset_yaw_tracking()
        print(f"  Reset yaw tracking for next controller")
        
        # Brief pause between controllers
        if controller_idx < len(controllers):
            print(f"\nWaiting 5 seconds before next controller...")
            rospy.sleep(5.0)
    
    # Save complete experiment summary
    all_results['experiment_info']['end_time'] = datetime.now().isoformat()
    
    # Save complete results (using the already created main experiment folder)
    complete_summary_path = os.path.join(main_experiment_folder, "complete_experiment_summary.json")
    with open(complete_summary_path, 'w') as f:
        import json
        json.dump(all_results, f, indent=2, default=str)
    
    # Final return to polygon (0,0) after completing all controllers
    print(f"\n--- Final return to polygon (0,0) after all controllers ---")
    
    # Move to higher position first (10mm higher than current position)
    print(f"  Moving to higher position (10mm) for safe return...")
    current_pose = rtde_help.getCurrentPose()
    high_pose = copy.deepcopy(current_pose)
    high_pose.pose.position.z = current_pose.pose.position.z + 10e-3  # 10mm higher
    rtde_help.goToPose_2Dhaptic(high_pose)
    rospy.sleep(0.5)
    
    # Move to polygon (0,0) position
    polygon_position = get_polygon_position(0, 0, base_disengage_position)
    
    # Set default yaw angle for return
    default_yaw = pi/2 + 45*pi/180  # Same as experiment default
    setOrientation = tf.transformations.quaternion_from_euler(default_yaw, pi, 0, 'szxy')
    return_pose = rtde_help.getPoseObj(polygon_position, setOrientation)
    
    print(f"  Moving to polygon (0,0) position...")
    print(f"  Position: X={polygon_position[0]*1000:.1f}mm, Y={polygon_position[1]*1000:.1f}mm, Z={polygon_position[2]*1000:.1f}mm")
    rtde_help.goToPose_2Dhaptic(return_pose)
    rospy.sleep(1.0)
    
    # Verify and correct orientation if needed
    current_pose = rtde_help.getCurrentPose()
    current_euler = tf.transformations.euler_from_quaternion([
        current_pose.pose.orientation.x, current_pose.pose.orientation.y, 
        current_pose.pose.orientation.z, current_pose.pose.orientation.w
    ], 'szxy')
    current_yaw = current_euler[0] * 180 / np.pi
    
    target_yaw = default_yaw * 180 / np.pi
    yaw_error = abs(current_yaw - target_yaw)
    
    if yaw_error > 5.0:  # If yaw error is more than 5 degrees
        print(f"  Correcting orientation: current yaw={current_yaw:.1f}°, target={target_yaw:.1f}°")
        # Move to the exact target orientation
        rtde_help.goToPose_2Dhaptic(return_pose)
        rospy.sleep(0.5)
    
    print(f"  Returned to polygon (0,0) position with correct orientation")
    
    print(f"\n{'='*60}")
    print("POLYGON GRID EXPERIMENT COMPLETED!")
    print(f"{'='*60}")
    print(f"Complete results saved to: {complete_summary_path}")
    print(f"Robot returned to polygon (0,0) position")
    
    # Print final comparison
    print(f"\nController Comparison:")
    for controller_name, results in all_results['controller_results'].items():
        stats = results['overall_stats']
        print(f"  {controller_name}: {stats['overall_success_rate']:.1f}% success, "
              f"{stats['boundary_failures']} boundary failures ({stats['boundary_failure_rate']:.1f}%), "
              f"{stats['avg_iterations']:.1f} avg iterations, "
              f"{stats['avg_path_length_2d']*1000:.1f}mm avg path")
    
    return all_results


def main(args):
    """
    Main function to setup and execute polygon grid experiments.
    """
    # Determine seeds to run
    if args.seeds is not None:
        seeds_to_run = args.seeds
        print(f"Running experiments for seeds: {seeds_to_run}")
    else:
        seeds_to_run = [None]  # Single run with random seed
        print("Using random seed (not reproducible)")
    
    # Initialize ROS node
    rospy.init_node('polygon_grid_experiment')
    
    # Setup helper functions (done once for all seeds)
    FT_help = FT_CallbackHelp()
    rospy.sleep(0.5)
    P_help = P_CallbackHelp()
    rospy.sleep(0.5)
    rtde_help = rtdeHelp(125)
    rospy.sleep(0.5)
    file_help = fileSaveHelp()
    search_help = hapticSearch2DHelp(P_vac=-20000, d_lat=1.0e-3, d_yaw=1.5, damping_factor=0.7, n_ch=args.ch, p_reverse=args.reverse)
    
    # Set TCP offset
    rospy.sleep(0.5)
    rtde_help.setTCPoffset([0, 0, 0.150, 0, 0, 0])
    if args.ch == 5:
        rtde_help.setTCPoffset([0, 0, 0.150 + 0.019, 0, 0, 0])
    if args.ch == 6:
        rtde_help.setTCPoffset([0, 0, 0.150 + 0.020, 0, 0, 0])
    rospy.sleep(0.2)
    
    # Set up publishers
    targetPWM_Pub = rospy.Publisher('pwm', Int8, queue_size=1)
    rospy.sleep(0.5)
    targetPWM_Pub.publish(0)  # Start with vacuum off
    
    # Set up data logging
    print("Wait for the data_logger to be enabled")
    rospy.wait_for_service('data_logging')
    dataLoggerEnable = rospy.ServiceProxy('data_logging', Enable)
    dataLoggerEnable(False)
    rospy.sleep(1)
    file_help.clearTmpFolder()
    
    try:
        input("Press <Enter> to start polygon experiment")
        
        if args.initial_contact_mode:
            # Run visual test mode (only once, doesn't use seeds)
            print("Running visual test mode...")
            run_polygon_visual_test(args, rtde_help)
            print("Polygon visual test completed successfully!")
        else:
            # Run polygon grid experiment for each seed
            all_results = []
            for i, seed in enumerate(seeds_to_run):
                print(f"\n{'='*60}")
                print(f"RUNNING EXPERIMENT {i+1}/{len(seeds_to_run)}")
                if seed is not None:
                    print(f"Seed: {seed}")
                    np.random.seed(seed)
                else:
                    print("Using random seed")
                print(f"{'='*60}")
                
                # Create a copy of args with the current seed
                current_args = copy.deepcopy(args)
                current_args.seed = seed
                
                # Run experiment for this seed
                results = run_polygon_grid_experiment(
                    current_args, rtde_help, search_help, P_help, targetPWM_Pub,
                    dataLoggerEnable, file_help, None, args.controllers
                )
                all_results.append(results)
                
                print(f"Completed experiment {i+1}/{len(seeds_to_run)}")
                if i < len(seeds_to_run) - 1:
                    print("Waiting before next experiment...")
                    rospy.sleep(2)  # Brief pause between experiments
            
            print(f"\n{'='*60}")
            print(f"ALL EXPERIMENTS COMPLETED!")
            print(f"Total experiments run: {len(seeds_to_run)}")
            print(f"{'='*60}")
        
    except rospy.ROSInterruptException:
        targetPWM_Pub.publish(0)
        return
    except KeyboardInterrupt:
        targetPWM_Pub.publish(0)
        return


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Polygon Grid Experiment for 2D Haptic Search')
    parser.add_argument('--ch', type=int, help='number of channels', default=4)
    parser.add_argument('--controllers', nargs='+', help='controllers to test', 
                       default=['greedy', 'rl_hyaw_momentum'])
    parser.add_argument('--max_iterations', type=int, help='maximum iterations per trial', default=50)
    parser.add_argument('--reverse', type=bool, help='use reverse airflow', default=False)
    parser.add_argument('--debug_mode', action='store_true', 
                       help='debug mode: run only 1 trial per polygon instead of 10 (still performs full haptic search)')
    parser.add_argument('--initial_contact_mode', action='store_true', 
                       help='visual test mode: test each polygon position by moving robot arm (no vacuum, no data saved)')
    parser.add_argument('--pause_time', type=float, help='pause time between experiments in seconds (default: 0.5s, try 0.2s for faster)', 
                      default=0.2)
    parser.add_argument('--hop_sleep', type=float, help='sleep time after hopping down in seconds (default: 0.08s, try 0.05s for faster)', 
                      default=0.2)
    parser.add_argument('--seeds', nargs='+', type=int, help='random seeds for reproducible experiments (e.g., --seeds 42 43 44). If not provided, uses random seed.', 
                      default=None)
    parser.add_argument('--difficulty', type=str, default='easy', help='difficulty mode: easy, medium, hard (default: easy)')
    parser.add_argument('--start_polygon', type=int, help='starting polygon number (1-12, where 1=polygon_0_0, 2=polygon_0_1, ..., 12=polygon_2_3)', 
                      default=1)
    
    args = parser.parse_args()
    
    # Validate start_polygon argument
    if args.start_polygon < 1 or args.start_polygon > 12:
        print(f"Error: start_polygon must be between 1 and 12, got {args.start_polygon}")
        exit(1)
    
    main(args)
