#!/usr/bin/env python

# Authors: Jungpyo Lee
# Create: Aug.26.2024
# Last update: Oct.11.2025
# Description: This script is used to test 2D haptic search models while recording pressure and path.
#              The difference between this script and JP_2D_haptic_search_continuous.py is that this script has a hopping motion.
#              This script will test heuristic controller and residual RL controller with hopping motion.
#              
#              For trap and dumbbell primitives, it supports two experiment modes:
#              - 'sweep' mode (default): Systematically sweeps 360° in 5° steps (72 experiments)
#              - 'random' mode: Tests 10 random yaw angles

# imports
try:
  import rospy
  import tf
  ros_enabled = True
except:
  print('Couldn\'t import ROS.  I assume you\'re running this on your laptop')
  ros_enabled = False

from calendar import month_abbr
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

def get_disEngagePosition(primitives):
  """
  Get the disengage position (safe position above the object) based on primitive type.
  This position is where the robot returns after each experiment.
  
  Args:
    primitives: Type of primitive shape ('trap', 'dumbbell', 'polygons', etc.)
  
  Returns:
    List of [x, y, z] coordinates in meters for the disengage position
  """
  if primitives == 'trap':
    disEngagePosition =  [0.443, -.038, 0.013]
  elif primitives == 'dumbbell':
    disEngagePosition =  [0.443 + 0.138, -.0395, 0.013]
  elif primitives == 'polygons':
    disEngagePosition =  [0.443, -.038, 0.013]
  else:
    # Default fallback for unknown primitives
    print(f"Warning: Unknown primitives '{primitives}', using trap position")
    disEngagePosition =  [0.443, -.038, 0.013]
  return disEngagePosition

def generate_random_yaw():
  """
  Generate random yaw angle for experimental trials.
  This is used to test different initial orientations.
  
  Returns:
    Random yaw angle between 0 and 2π radians
  """
  return np.random.uniform(0, 2 * np.pi)

def return_to_original_yaw(rtde_help, original_disEngagePose, total_yaw_rotated, yaw_step_deg):
  """
  Return the end effector to its original yaw position after a yaw sweep experiment.
  
  This function implements the key insight that to return to the original position:
  1. Reverse the yaw direction (move in opposite direction)
  2. Divide the total angle by 3 and execute in three separate movements
  3. Each movement rotates -120 degrees (for a 360° sweep)
  
  Args:
    rtde_help: RTDE helper for robot control
    original_disEngagePose: The original disengage pose before the sweep
    total_yaw_rotated: Total yaw angle rotated during the sweep (in radians)
    yaw_step_deg: Yaw step size used in the sweep (in degrees)
  
  Returns:
    Final yaw angle in degrees after all three movements
  """
  print(f"Returning to original yaw position...")
  print(f"Total yaw rotated during sweep: {total_yaw_rotated * 180.0 / np.pi:.1f}°")
  
  # Calculate return angle per movement: reverse direction and divide by 3
  return_angle_per_move = -total_yaw_rotated / 3.0
  return_angle_per_move_deg = return_angle_per_move * 180.0 / np.pi
  
  print(f"Return angle per movement (reversed and divided by 3): {return_angle_per_move_deg:.1f}°")
  print(f"Executing 3 separate movements of {return_angle_per_move_deg:.1f}° each...")
  
  # Get current pose for the first movement
  current_pose = rtde_help.getCurrentPose()
  
  # Execute three separate movements
  for move_num in range(1, 4):
    print(f"--- Movement {move_num}/3 ---")
    
    # Get current pose and calculate new yaw
    q = current_pose.pose.orientation
    current_euler = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w], 'szxy')
    
    # Apply the return angle for this movement
    new_yaw = current_euler[0] - return_angle_per_move  # Subtract to change rotation direction
    setOrientation = tf.transformations.quaternion_from_euler(new_yaw, current_euler[1], current_euler[2], 'szxy')
    
    # Create new pose with corrected yaw
    return_pose = copy.deepcopy(current_pose)
    return_pose.pose.orientation.x = setOrientation[0]
    return_pose.pose.orientation.y = setOrientation[1]
    return_pose.pose.orientation.z = setOrientation[2]
    return_pose.pose.orientation.w = setOrientation[3]
    
    # Move to the corrected yaw position
    print(f"Moving {return_angle_per_move_deg:.1f}° (movement {move_num}/3)...")
    rtde_help.goToPose(return_pose)
    rospy.sleep(0.5)  # Brief pause to allow movement to complete
    
    # Update current_pose for next iteration
    current_pose = rtde_help.getCurrentPose()
    
    # Show progress
    q_current = current_pose.pose.orientation
    current_euler = tf.transformations.euler_from_quaternion([q_current.x, q_current.y, q_current.z, q_current.w], 'szxy')
    current_yaw_deg = current_euler[0] * 180.0 / np.pi
    print(f"Current yaw after movement {move_num}: {current_yaw_deg:.1f}°")
  
  # Verify the final yaw angle
  final_pose = rtde_help.getCurrentPose()
  q_final = final_pose.pose.orientation
  final_euler = tf.transformations.euler_from_quaternion([q_final.x, q_final.y, q_final.z, q_final.w], 'szxy')
  final_yaw_deg = final_euler[0] * 180.0 / np.pi
  
  print(f"Return to original yaw completed. Final yaw: {final_yaw_deg:.1f}°")
  
  return final_yaw_deg

def run_multi_controller_yaw_sweep(args, rtde_help, search_help, P_help, targetPWM_Pub, 
                                  dataLoggerEnable, file_help, disEngagePose, rl_controller):
  """
  Run yaw sweep experiments with multiple controllers in sequence.
  
  This function automatically tests all four heuristic controllers:
  - greedy
  - yaw  
  - momentum
  - yaw_momentum
  
  For each controller, it runs the complete yaw sweep experiment and returns
  to the original yaw position before moving to the next controller.
  
  Args:
    args: Command line arguments containing experiment parameters
    rtde_help: RTDE helper for robot control
    search_help: Helper for 2D haptic search computations
    P_help: Helper for pressure sensor data
    targetPWM_Pub: ROS publisher for vacuum pump PWM control
    dataLoggerEnable: ROS service proxy for enabling/disabling data logging
    file_help: Helper for saving experiment data
    disEngagePose: Safe position above the object (fixed for all experiments)
    rl_controller: RL controller object (if using RL-based control)
  
  Returns:
    Dictionary containing results from all controllers
  """
  
  # Define the list of controllers to test
  controllers_to_test = ['momentum', 'yaw_momentum']
  
  print(f"Starting multi-controller yaw sweep experiment")
  print(f"Controllers to test: {', '.join(controllers_to_test)}")
  print(f"Primitives: {args.primitives}")
  print(f"Yaw step: {args.yaw_step}°")
  print("=" * 60)
  
  # Store original disEngagePose for each controller
  original_disEngagePose = copy.deepcopy(disEngagePose)
  
  # Results storage for all controllers
  all_controller_results = {}
  
  # Run experiments for each controller
  for controller_idx, controller_name in enumerate(controllers_to_test, 1):
    print(f"\n{'='*20} CONTROLLER {controller_idx}/4: {controller_name.upper()} {'='*20}")
    
    # Update args with current controller
    original_controller = args.controller
    args.controller = controller_name
    
    # Create controller-specific folder name within the date folder
    controller_folder_name = f"{args.primitives}_ch{args.ch}_{controller_name}"
    print(f"Creating folder: {controller_folder_name}")
    
    # Create a new file_help instance with controller-specific folder
    controller_file_help = fileSaveHelp(savingFolderName=controller_folder_name)
    
    # Modify the ResultSavingDirectory to include date folder first
    from datetime import datetime
    date_folder = datetime.now().strftime("%y%m%d")
    base_folder = os.path.expanduser('~') + '/SuctionExperiment'
    controller_file_help.ResultSavingDirectory = os.path.join(base_folder, date_folder, controller_folder_name)
    
    # Create the directory if it doesn't exist
    if not os.path.exists(controller_file_help.ResultSavingDirectory):
      os.makedirs(controller_file_help.ResultSavingDirectory)
    
    print(f"Data will be saved to: {controller_file_help.ResultSavingDirectory}")
    
    # Create a fresh disEngagePose for this controller
    current_disEngagePose = copy.deepcopy(original_disEngagePose)
    
    # Run the yaw sweep experiment for this controller
    print(f"Starting yaw sweep with {controller_name} controller...")
    
    try:
      # Call the existing run_repeated_experiments function with controller-specific file_help
      run_repeated_experiments(args, rtde_help, search_help, P_help, targetPWM_Pub,
                              dataLoggerEnable, controller_file_help, current_disEngagePose, rl_controller, 
                              mode='sweep')
      
      print(f"✓ {controller_name} controller completed successfully")
      
    except Exception as e:
      print(f"✗ {controller_name} controller failed: {e}")
      all_controller_results[controller_name] = {
        'status': 'failed',
        'error': str(e),
        'folder': controller_file_help.ResultSavingDirectory
      }
      continue
    
    # Store results for this controller
    all_controller_results[controller_name] = {
      'status': 'completed',
      'controller_index': controller_idx,
      'folder': controller_file_help.ResultSavingDirectory
    }
    
    # Brief pause between controllers
    if controller_idx < len(controllers_to_test):
      print(f"\nWaiting 2 seconds before next controller...")
      rospy.sleep(2.0)
    
    # Restore original controller for next iteration
    args.controller = original_controller
  
  # Print final summary
  print(f"\n{'='*60}")
  print("MULTI-CONTROLLER EXPERIMENT SUMMARY")
  print(f"{'='*60}")
  
  completed_controllers = [name for name, result in all_controller_results.items() 
                          if result['status'] == 'completed']
  failed_controllers = [name for name, result in all_controller_results.items() 
                       if result['status'] == 'failed']
  
  print(f"Total controllers tested: {len(controllers_to_test)}")
  print(f"Successfully completed: {len(completed_controllers)}")
  print(f"Failed: {len(failed_controllers)}")
  
  if completed_controllers:
    print(f"Completed controllers: {', '.join(completed_controllers)}")
    print(f"\nData saved in the following folders:")
    for controller_name in completed_controllers:
      if controller_name in all_controller_results:
        folder_path = all_controller_results[controller_name].get('folder', 'Unknown')
        print(f"  {controller_name}: {folder_path}")
  
  if failed_controllers:
    print(f"Failed controllers: {', '.join(failed_controllers)}")
    for controller_name in failed_controllers:
      if controller_name in all_controller_results:
        folder_path = all_controller_results[controller_name].get('folder', 'Unknown')
        print(f"  {controller_name}: {folder_path}")
  
  print(f"{'='*60}")
  
  return all_controller_results

def run_haptic_search_loop(args, rtde_help, search_help, P_help, targetPWM_Pub, 
                          dataLoggerEnable, file_help, disEngagePose, rl_controller, 
                          experiment_num=None, random_yaw=None):
  """
  Run the main haptic search loop with hopping motion.
  
  This is the core function that performs the 2D haptic search with hopping motion.
  It handles:
    1. Starting data logging and vacuum pump
    2. Moving to engage position (5mm below disengage position)
    3. Running the haptic search control loop until success or max iterations
    4. Stopping data logging and vacuum pump
  
  The function uses either heuristic controllers (greedy, yaw, momentum, etc.) or
  RL-based controllers to compute movements based on pressure readings.
  
  Args:
    args: Command line arguments containing experiment parameters
    rtde_help: Helper for robot control via RTDE interface
    search_help: Helper for 2D haptic search computations
    P_help: Helper for pressure sensor data
    targetPWM_Pub: ROS publisher for vacuum pump PWM control
    dataLoggerEnable: ROS service proxy for enabling/disabling data logging
    file_help: Helper for saving experiment data
    disEngagePose: Safe position above the object (starting pose)
    rl_controller: RL controller object (if using RL-based control)
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
  # Set up constants for vacuum pump PWM and success criteria
  DUTYCYCLE_100 = 100  # Full vacuum power
  DUTYCYCLE_0 = 0      # No vacuum
  P_vac = search_help.P_vac  # Vacuum pressure threshold for success
  max_steps = args.max_iterations  # Maximum iterations before giving up
  
  # ==================== START EXPERIMENT ====================
  # Start vacuum pump at full power and begin data collection
  targetPWM_Pub.publish(DUTYCYCLE_100)
  P_help.startSampling()      
  rospy.sleep(1)
  P_help.setNowAsOffset()  # Set current pressure readings as baseline offset
  dataLoggerEnable(True)   # Enable data logging (pressure, force, position, etc.)

  # ==================== MOVE TO ENGAGE POSITION ====================
  # First, rotate to the new yaw at the disEngagePose height (before moving down)
  rtde_help.goToPose(disEngagePose)
  rospy.sleep(0.2)  # Brief pause after rotation
  
  # Then move straight down to engage position (5mm below)
  # This is the starting height for haptic search with contact to the object
  engagePosition = copy.deepcopy(disEngagePose.pose.position)
  engagePosition.z = engagePosition.z - 5e-3  # Lower by 5mm (critical for proper seal formation)
  engagePose = rtde_help.getPoseObj(engagePosition, disEngagePose.pose.orientation)
  rtde_help.goToPose_2Dhaptic(engagePose)
  
  # ==================== INITIALIZE CONTROL LOOP VARIABLES ====================
  suctionSuccessFlag = False  # Becomes True when vacuum seal is achieved
  startTime = time.time()     # For computing elapsed time
  iteration_count = 1         # Control iteration counter (starts at 1 for first check)
  pose_diff = 0               # Tracks Z-position error for compensation
  orientation_error = [0, 0, 0]  # Tracks pitch/roll error for compensation [yaw_err, pitch_err, roll_err]
  
  # Boundary limits (in world frame)
  initial_x_position = disEngagePose.pose.position.x  # Store initial X position
  initial_y_position = disEngagePose.pose.position.y  # Store initial Y position
  
  # Set boundary limit for non-trap objects (trap uses hardcoded asymmetric limits)
  if args.primitives == 'trap':
    boundary_limit = None  # Trap uses hardcoded asymmetric limits in boundary check
  else:
    boundary_limit = 15e-3  # 15mm boundary limit for other objects
  
  boundary_exceeded = False  # Flag for boundary violation
  
  # Determine boundary type based on primitive
  # Trap and dumbbell: Y-axis only (elongated along X)
  # Polygons: Circular 2D boundary
  use_circular_boundary = args.primitives not in ['trap', 'dumbbell']
  
  # Path length tracking
  path_length_2d = 0.0        # Total 2D path length (XY plane) in meters
  path_length_3d = 0.0        # Total 3D path length (XYZ) in meters
  prev_position = None        # Previous position for path length calculation
  
  # Reset RL controller history if using RL-based control
  # This clears any previous state information
  if rl_controller and rl_controller.is_model_loaded():
    rl_controller.reset_history()
  
  # ==================== MAIN HAPTIC SEARCH CONTROL LOOP ====================
  while not suctionSuccessFlag:
    iteration_start_time = time.time()
    
    # ===== CHECK SUCCESS CONDITION =====
    # Read pressure from all chambers and check if vacuum seal is formed
    P_array = P_help.four_pressure
    reached_vacuum = all(np.array(P_array) < P_vac)  # All chambers below vacuum threshold?
    # print(f"P_array: {P_array}")
    if reached_vacuum:
      # SUCCESS! Vacuum seal achieved
      suctionSuccessFlag = True
      args.elapsedTime = time.time() - startTime
      args.iteration_count = iteration_count
      if experiment_num:
        print(f"Experiment {experiment_num}: SUCCESS at iteration {iteration_count}")
      else:
        print(f"Suction engage succeeded with controller {args.controller} at iteration {iteration_count}")
      rtde_help.stopAtCurrPoseAdaptive()  # Stop at successful position
      rospy.sleep(1)  # Hold position to ensure data is recorded
      break
    
    # ===== CHECK FAILURE CONDITION =====
    # If max iterations reached without success, abort experiment
    elif iteration_count >= max_steps:
      args.timeOverFlag = True
      args.elapsedTime = time.time() - startTime
      suctionSuccessFlag = False
      if experiment_num:
        print(f"Experiment {experiment_num}: FAILED - Max iterations reached")
      else:
        print(f"Suction controller failed! Reached maximum iterations ({max_steps})")
      rtde_help.stopAtCurrPoseAdaptive()
      targetPWM_Pub.publish(DUTYCYCLE_0)  # Turn off vacuum
      rospy.sleep(0.1)
      break

    # ===== GET CURRENT STATE =====
    # Read current robot pose and compute yaw angle for controller
    measuredCurrPose = rtde_help.getCurrentPose()
    T_curr = search_help.get_Tmat_from_Pose(measuredCurrPose)
    
    # ===== CHECK BOUNDARY CONDITION =====
    # Different boundary checks based on primitive type
    current_x = measuredCurrPose.pose.position.x
    current_y = measuredCurrPose.pose.position.y
    
    if use_circular_boundary:
      # Polygons: Check 2D circular boundary (yaw-invariant)
      dx = current_x - initial_x_position
      dy = current_y - initial_y_position
      displacement = np.sqrt(dx**2 + dy**2)
      boundary_type_str = "2D circular"
      boundary_exceeded = displacement > boundary_limit
    elif args.primitives == 'trap':
      # Trap: Check both X and Y axes separately with asymmetric X limits
      dx = current_x - initial_x_position  # Can be positive or negative
      dy = abs(current_y - initial_y_position)
      
      # Check X-axis with asymmetric limits
      if dx > 0:  # Positive X direction
        x_exceeded = dx > 25e-3  # 25mm limit for positive X
        x_limit = 20.0
      else:  # Negative X direction
        x_exceeded = abs(dx) > 8e-3  # 8mm limit for negative X
        x_limit = 8.0
      
      y_exceeded = dy > 15e-3  # 15mm Y-axis limit
      boundary_exceeded = x_exceeded or y_exceeded
      
      if x_exceeded:
        boundary_type_str = "X-axis"
        displacement = abs(dx)
        x_direction = "positive" if dx > 0 else "negative"
      elif y_exceeded:
        boundary_type_str = "Y-axis"
        displacement = dy
        x_direction = None
      else:
        boundary_type_str = "None"
        displacement = 0
        x_direction = None
    else:
      # Dumbbell: Check Y-axis only (objects are elongated along X)
      displacement = abs(current_y - initial_y_position)
      boundary_type_str = "Y-axis"
      boundary_exceeded = displacement > boundary_limit
    
    if boundary_exceeded:
      args.timeOverFlag = True
      args.elapsedTime = time.time() - startTime
      suctionSuccessFlag = False
      if experiment_num:
        if args.primitives == 'trap':
          if boundary_type_str == "X-axis":
            print(f"Experiment {experiment_num}: FAILED - {boundary_type_str} boundary exceeded ({displacement*1000:.1f}mm > {x_limit:.1f}mm, {x_direction} direction)")
          else:  # Y-axis
            print(f"Experiment {experiment_num}: FAILED - {boundary_type_str} boundary exceeded ({displacement*1000:.1f}mm > 15.0mm)")
        else:
          print(f"Experiment {experiment_num}: FAILED - {boundary_type_str} boundary exceeded ({displacement*1000:.1f}mm > {boundary_limit*1000:.1f}mm)")
      else:
        if args.primitives == 'trap':
          if boundary_type_str == "X-axis":
            print(f"Suction controller failed! {boundary_type_str} boundary exceeded ({displacement*1000:.1f}mm > {x_limit:.1f}mm, {x_direction} direction)")
          else:  # Y-axis
            print(f"Suction controller failed! {boundary_type_str} boundary exceeded ({displacement*1000:.1f}mm > 15.0mm)")
        else:
          print(f"Suction controller failed! {boundary_type_str} boundary exceeded ({displacement*1000:.1f}mm > {boundary_limit*1000:.1f}mm)")
      rtde_help.stopAtCurrPoseAdaptive()
      targetPWM_Pub.publish(DUTYCYCLE_0)  # Turn off vacuum
      rospy.sleep(0.1)
      break
    yaw_angle = convert_yawAngle(search_help.get_yawRotation_from_T(T_curr))

    # ===== COMPUTE CONTROL ACTION =====
    # Use either RL controller or heuristic controller based on configuration
    if rl_controller and rl_controller.is_model_loaded():
      # RL-BASED CONTROL: Neural network computes action from pressure and yaw
      try:
        lateral_vel, yaw_vel, debug_info = rl_controller.compute_action(P_array, yaw_angle, return_debug=True)
        
        # Convert RL output (velocities) to transformation matrices
        step_lateral = search_help.d_lat  # Lateral step size (mm)
        step_yaw = search_help.d_yaw      # Yaw step size (degrees)
        
        # Create lateral movement transformation in body frame
        dx_lat = lateral_vel[0] * step_lateral
        dy_lat = lateral_vel[1] * step_lateral
        T_later = search_help.get_Tmat_TranlateInBodyF([dx_lat, dy_lat, 0.0])
        
        # Create yaw rotation transformation
        if abs(yaw_vel) > 1e-6:  # Only rotate if significant yaw command
          d_yaw_rad = yaw_vel * step_yaw * np.pi / 180.0
          rot_axis = np.array([0, 0, -1])  # Rotate around Z-axis
          omega_hat = hat(rot_axis)
          Rw = scipy.linalg.expm(d_yaw_rad * omega_hat)
          T_yaw = create_transform_matrix(Rw, [0, 0, 0])
        else:
          T_yaw = np.eye(4)  # No rotation
        
        T_align = np.eye(4)  # No alignment for RL
        
        # Print debug info occasionally
        if iteration_count % 50 == 0:
          print(f"RL Controller - Lateral: {lateral_vel}, Yaw: {yaw_vel:.3f}")
          print(f"Debug: {debug_info}")
          
      except Exception as e:
        # Fallback to greedy heuristic if RL fails
        print(f"RL controller failed: {e}, falling back to heuristic")
        T_later, T_yaw, T_align = search_help.get_Tmats_from_controller(P_array, "greedy")
    else:
      # HEURISTIC CONTROL: Rule-based controller (greedy, yaw, momentum, etc.)
      T_later, T_yaw, T_align = search_help.get_Tmats_from_controller(P_array, args.controller)
    
    # Combine transformations: lateral → yaw → align
    T_move = T_later @ T_yaw @ T_align

    # ===== EXECUTE HOPPING MOTION =====
    # Apply computed transformation with hopping to avoid friction and improve contact
    measuredCurrPose = rtde_help.getCurrentPose()
    currPose = search_help.get_PoseStamped_from_T_initPose(T_move, measuredCurrPose)
    
    # Apply orientation compensation to correct for pitch/roll drift from contact forces
    q = currPose.pose.orientation
    curr_euler = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w], 'szxy')
    compensated_euler = [curr_euler[0] - orientation_error[0],  # yaw (usually keep as is)
                        curr_euler[1] - orientation_error[1],  # pitch compensation
                        curr_euler[2] - orientation_error[2]]  # roll compensation
    compensated_quat = tf.transformations.quaternion_from_euler(compensated_euler[0], compensated_euler[1], compensated_euler[2], 'szxy')
    currPose.pose.orientation.x = compensated_quat[0]
    currPose.pose.orientation.y = compensated_quat[1]
    currPose.pose.orientation.z = compensated_quat[2]
    currPose.pose.orientation.w = compensated_quat[3]
    
    # HOP UP: Lift 5mm to disengage from surface
    currPose.pose.position.z = currPose.pose.position.z + 5e-3
    rtde_help.goToPose_2Dhaptic(currPose)
    # rospy.sleep(0.02)  # Brief pause at top of hop
    
    # HOP DOWN: Return to contact, compensating for any Z-position error
    currPose.pose.position.z = currPose.pose.position.z - 5e-3 - pose_diff
    rtde_help.goToPose_2Dhaptic(currPose)
    rospy.sleep(args.hop_sleep)  # Brief pause at bottom of hop (configurable)
    
    # ===== UPDATE STATE =====
    # Measure actual position and update error compensation
    measuredCurrPose = rtde_help.getCurrentPose()
    T_curr = search_help.get_Tmat_from_Pose(measuredCurrPose)
    args.final_yaw = convert_yawAngle(search_help.get_yawRotation_from_T(T_curr))
    pose_diff = measuredCurrPose.pose.position.z - currPose.pose.position.z  # Z-error for next iteration
    
    # Update orientation error for next iteration (similar to pose_diff for Z)
    q_measured = measuredCurrPose.pose.orientation
    q_commanded = currPose.pose.orientation
    measured_euler = tf.transformations.euler_from_quaternion([q_measured.x, q_measured.y, q_measured.z, q_measured.w], 'szxy')
    commanded_euler = tf.transformations.euler_from_quaternion([q_commanded.x, q_commanded.y, q_commanded.z, q_commanded.w], 'szxy')
    orientation_error = [measured_euler[i] - commanded_euler[i] for i in range(3)]  # [yaw, pitch, roll] error
    
    # ===== UPDATE PATH LENGTH =====
    # Calculate cumulative path length traveled
    curr_position = measuredCurrPose.pose.position
    if prev_position is not None:
      # Calculate 2D path length (XY plane only)
      dx = curr_position.x - prev_position.x
      dy = curr_position.y - prev_position.y
      path_length_2d += np.sqrt(dx**2 + dy**2)
      
      # Calculate 3D path length (XYZ)
      dz = curr_position.z - prev_position.z
      path_length_3d += np.sqrt(dx**2 + dy**2 + dz**2)
    
    prev_position = curr_position

    # Increment iteration counter
    iteration_count += 1
    iteration_end_time = time.time()

    # ===== PROGRESS MONITORING =====
    # Print progress every 10 iterations
    if iteration_count % 10 == 0:
        if experiment_num:
          print(f"Experiment {experiment_num} - Iteration {iteration_count}/{max_steps}")
        else:
          print(f"Iteration {iteration_count}/{max_steps}")

     # Measure and display control frequency every 100 iterations
    if iteration_count % 100 == 0:
        current_time = time.time()
        elapsed_time = current_time - startTime
        frequency = iteration_count / elapsed_time
        if experiment_num:
          print(f"Experiment {experiment_num} - Current control frequency after {iteration_count} iterations: {frequency:.1f} Hz")
        else:
          print(f"Current control frequency after {iteration_count} iterations: {frequency:.1f} Hz")

  # ==================== CLEANUP AND FINALIZATION ====================
  # Stop data collection and turn off vacuum pump
  rospy.sleep(0.2)
  dataLoggerEnable(False)  # Stop data logging (data will be saved by file_help)
  rospy.sleep(0.2)
  P_help.stopSampling()    # Stop pressure sampling
  targetPWM_Pub.publish(DUTYCYCLE_0)  # Turn off vacuum pump

  # Store results in args for data saving
  args.suction = suctionSuccessFlag
  args.iteration_count = iteration_count
  args.path_length_2d = path_length_2d
  args.path_length_3d = path_length_3d
  args.boundary_exceeded = boundary_exceeded
  
  return {
    'success': suctionSuccessFlag,
    'iterations': iteration_count,
    'elapsed_time': args.elapsedTime if hasattr(args, 'elapsedTime') else 0,
    'final_yaw': args.final_yaw if hasattr(args, 'final_yaw') else 0,
    'path_length_2d': path_length_2d,
    'path_length_3d': path_length_3d,
    'boundary_exceeded': boundary_exceeded
  }

def run_single_experiment(args, rtde_help, search_help, P_help, targetPWM_Pub, 
                         dataLoggerEnable, file_help, disEngagePose, rl_controller, 
                         experiment_num, random_yaw, tracking_yaw=None):
  """
  Run a single haptic search experiment with a specific random yaw angle.
  
  This function is called by run_repeated_experiments() for each trial.
  It performs the following steps:
    1. Adds random yaw to the base orientation from disEngagePose
    2. Calls run_haptic_search_loop() to execute the experiment
    3. Saves the experiment data to file with unique filename
    4. Returns to disengage position
  
  The saved data includes experiment number and yaw angle in the filename for
  easy identification (e.g., exp01_yaw45.3deg).
  
  Args:
    args: Command line arguments containing experiment parameters
    rtde_help: Helper for robot control via RTDE interface
    search_help: Helper for 2D haptic search computations
    P_help: Helper for pressure sensor data
    targetPWM_Pub: ROS publisher for vacuum pump PWM control
    dataLoggerEnable: ROS service proxy for enabling/disabling data logging
    file_help: Helper for saving experiment data
    disEngagePose: Safe position above the object (with base orientation)
    rl_controller: RL controller object (if using RL-based control)
    experiment_num: Experiment number in the sequence (1, 2, 3, ...)
    random_yaw: Random yaw angle in radians to add to base orientation
  
  Returns:
    Dictionary containing:
      - experiment_num: Experiment number
      - success: Boolean indicating if suction was achieved
      - iterations: Number of control iterations executed
      - elapsed_time: Total time for the experiment
      - random_yaw: The random yaw angle used (radians)
      - final_yaw: Final yaw angle at end of experiment
      - path_length_2d: Total 2D path length (XY plane) in meters
      - path_length_3d: Total 3D path length (XYZ) in meters
  """
  
  # Get the current orientation from disEngagePose and add random yaw
  # Extract current euler angles from the disEngagePose
  q = disEngagePose.pose.orientation
  current_euler = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w], 'szxy')
  
  # Add random yaw to the existing yaw angle (subtract to reverse direction)
  new_yaw = current_euler[0] - random_yaw
  setOrientation = tf.transformations.quaternion_from_euler(new_yaw, current_euler[1], current_euler[2], 'szxy')
  
  # Create a new disEngagePose with the random yaw for this experiment
  disEngagePose_randomYaw = copy.deepcopy(disEngagePose)
  disEngagePose_randomYaw.pose.orientation.x = setOrientation[0]
  disEngagePose_randomYaw.pose.orientation.y = setOrientation[1]
  disEngagePose_randomYaw.pose.orientation.z = setOrientation[2]
  disEngagePose_randomYaw.pose.orientation.w = setOrientation[3]
  
  # Use tracking_yaw for filename and results if provided (for sweep mode)
  # Otherwise use random_yaw (for random mode)
  yaw_for_tracking = tracking_yaw if tracking_yaw is not None else random_yaw
  
  print(f"Experiment {experiment_num}: Yaw = {yaw_for_tracking*180/np.pi:.1f} degrees")
  
  # Run the haptic search using the common function
  result = run_haptic_search_loop(args, rtde_help, search_help, P_help, targetPWM_Pub, 
                                 dataLoggerEnable, file_help, disEngagePose_randomYaw, rl_controller, 
                                 experiment_num, yaw_for_tracking)

  # Save data for this experiment
  args.experiment_num = experiment_num
  args.random_yaw = yaw_for_tracking
  file_help.saveDataParams(args, appendTxt=f'jp_2D_hopping_ch_{args.ch}_{args.primitives}_controller_{args.controller}_exp{experiment_num:02d}_yaw{yaw_for_tracking*180/np.pi:.1f}deg')
  file_help.clearTmpFolder()
  
  # Return to disengage pose (first go 5mm higher, then to actual disEngagePose)
  disEngagePose_high = copy.deepcopy(disEngagePose)
  disEngagePose_high.pose.position.z = disEngagePose.pose.position.z + 5e-3  # 5mm higher
  rtde_help.goToPose(disEngagePose_high)  # Go to higher position first
  rospy.sleep(0.2)  # Brief pause
  rtde_help.goToPose(disEngagePose)  # Then go to actual disEngagePose
  
  return {
    'experiment_num': experiment_num,
    'success': result['success'],
    'iterations': result['iterations'],
    'elapsed_time': result['elapsed_time'],
    'random_yaw': yaw_for_tracking,
    'final_yaw': result['final_yaw'],
    'path_length_2d': result['path_length_2d'],
    'path_length_3d': result['path_length_3d']
  }

def run_repeated_experiments(args, rtde_help, search_help, P_help, targetPWM_Pub, 
                            dataLoggerEnable, file_help, disEngagePose, rl_controller, mode='sweep'):
  """
  Run repeated haptic search experiments with different yaw orientations.
  
  This function is specifically designed for trap and dumbbell primitives where
  we want to test the same fixed starting position with different yaw angles.
  
  Two modes are available:
    - 'sweep': Systematically sweep through 360 degrees in fixed increments (default)
               Example: 0°, 5°, 10°, ..., 355° (72 experiments total)
    - 'random': Use random yaw angles for each experiment (10 experiments)
  
  Experimental procedure:
    1. Validates that primitive type is 'trap' or 'dumbbell'
    2. Generates yaw angles based on selected mode
    3. Runs experiment for each yaw angle
    4. Collects results from all experiments
    5. Computes and displays summary statistics (success rate, avg iterations, etc.)
    6. Saves individual experiment data files and a summary JSON file
  
  The function ensures a 2-second pause between experiments to allow the robot
  and sensors to stabilize.
  
  Args:
    args: Command line arguments containing experiment parameters
    rtde_help: Helper for robot control via RTDE interface
    search_help: Helper for 2D haptic search computations
    P_help: Helper for pressure sensor data
    targetPWM_Pub: ROS publisher for vacuum pump PWM control
    dataLoggerEnable: ROS service proxy for enabling/disabling data logging
    file_help: Helper for saving experiment data
    disEngagePose: Safe position above the object (fixed for all experiments)
    rl_controller: RL controller object (if using RL-based control)
    mode: Experiment mode - 'sweep' (default) or 'random'
  
  Returns:
    None (saves results to files and prints summary statistics)
  """
  
  if args.primitives not in ['trap', 'dumbbell']:
    print(f"Repeated experiments only supported for 'trap' and 'dumbbell', got '{args.primitives}'")
    return
  
  # ==================== SETUP EXPERIMENT PARAMETERS ====================
  # Store original disEngagePose for return-to-original-yaw functionality
  original_disEngagePose = copy.deepcopy(disEngagePose)
  
  if mode == 'sweep':
    # YAW SWEEP MODE: Systematically sweep through 360 degrees
    yaw_step_deg = args.yaw_step  # Step size in degrees (configurable)
    num_experiments = int(360 / yaw_step_deg)  # Number of experiments based on step size
    yaw_step_rad = yaw_step_deg * np.pi / 180.0  # Incremental step in radians
    mode_description = f"yaw sweep (0° to {360-yaw_step_deg:.0f}° in {yaw_step_deg:.0f}° steps, {num_experiments} experiments)"
    
    # Handle resuming from a specific experiment
    start_exp = args.start_experiment
    if start_exp > 1:
      print(f"RESUMING from experiment {start_exp}")
      # Calculate the cumulative yaw for the starting experiment
      # Experiment 1 is at yaw 0°, experiment 2 at yaw_step_deg, etc.
      cumulative_yaw = (start_exp - 1) * yaw_step_rad
      
      # Update disEngagePose to the correct starting yaw
      q = disEngagePose.pose.orientation
      current_euler = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w], 'szxy')
      new_yaw = current_euler[0] - cumulative_yaw  # Apply the cumulative yaw
      setOrientation = tf.transformations.quaternion_from_euler(new_yaw, current_euler[1], current_euler[2], 'szxy')
      disEngagePose.pose.orientation.x = setOrientation[0]
      disEngagePose.pose.orientation.y = setOrientation[1]
      disEngagePose.pose.orientation.z = setOrientation[2]
      disEngagePose.pose.orientation.w = setOrientation[3]
      print(f"Starting yaw angle: {cumulative_yaw * 180.0 / np.pi:.1f}°")
    else:
      cumulative_yaw = 0.0
  else:
    # RANDOM YAW MODE: Random yaw angles for each experiment
    num_experiments = 10
    yaw_angles = [generate_random_yaw() for _ in range(num_experiments)]
    mode_description = "random yaw angles"
    start_exp = args.start_experiment if hasattr(args, 'start_experiment') else 1
    cumulative_yaw = 0.0
  
  results = []
  
  print(f"Starting {num_experiments} repeated experiments for {args.primitives}")
  print(f"Mode: {mode_description}")
  if start_exp > 1:
    print(f"Resuming from experiment {start_exp} to {num_experiments}")
  print("=" * 50)
  
  # ==================== RUN EXPERIMENTS ====================
  for i in range(start_exp - 1, num_experiments):  # Start from start_exp-1 (0-indexed)
    if mode == 'sweep':
      # For sweep mode: yaw is already in disEngagePose, so pass 0 (no additional yaw)
      # cumulative_yaw tracks the total yaw for display purposes
      yaw_angle = 0  # No additional yaw (already in disEngagePose)
      yaw_increment = yaw_step_rad
      display_yaw = cumulative_yaw  # For display only
    else:
      # For random mode, use the pre-generated random angles
      yaw_angle = yaw_angles[i]
      yaw_increment = 0  # No increment needed for random mode
      display_yaw = yaw_angle
    
    yaw_deg = display_yaw * 180.0 / np.pi
    
    print(f"\n--- Experiment {i+1}/{num_experiments} (Yaw: {yaw_deg:.1f}°) ---")
    
    # Run single experiment
    result = run_single_experiment(
      args, rtde_help, search_help, P_help, targetPWM_Pub,
      dataLoggerEnable, file_help, disEngagePose, rl_controller,
      i+1, yaw_angle, tracking_yaw=display_yaw
    )
    
    results.append(result)
    
    # Display running statistics
    successful_so_far = len([r for r in results if r['success']])
    success_rate_so_far = successful_so_far / len(results) * 100
    print(f"Running stats: {successful_so_far}/{len(results)} successful ({success_rate_so_far:.1f}% success rate)")
    if successful_so_far > 0:
      avg_iterations_so_far = np.mean([r['iterations'] for r in results if r['success']])
      print(f"Average iterations (successful so far): {avg_iterations_so_far:.1f}")
    
    # Update disEngagePose with new yaw for next experiment (sweep mode only)
    if mode == 'sweep' and i < num_experiments - 1:
      cumulative_yaw += yaw_increment
      # Apply incremental yaw to disEngagePose for next experiment
      q = disEngagePose.pose.orientation
      current_euler = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w], 'szxy')
      new_yaw = current_euler[0] - yaw_increment  # Apply increment (subtract for correct direction)
      setOrientation = tf.transformations.quaternion_from_euler(new_yaw, current_euler[1], current_euler[2], 'szxy')
      disEngagePose.pose.orientation.x = setOrientation[0]
      disEngagePose.pose.orientation.y = setOrientation[1]
      disEngagePose.pose.orientation.z = setOrientation[2]
      disEngagePose.pose.orientation.w = setOrientation[3]
    
    # Brief pause between experiments
    if i < num_experiments - 1:
      print(f"Waiting {args.pause_time:.1f} seconds before next experiment...")
      rospy.sleep(args.pause_time)
  
  # Print summary statistics
  print("\n" + "=" * 50)
  print("EXPERIMENT SUMMARY")
  print("=" * 50)
  
  successful_experiments = [r for r in results if r['success']]
  boundary_failures = [r for r in results if r.get('boundary_exceeded', False)]
  success_rate = len(successful_experiments) / len(results) * 100
  
  print(f"Total experiments: {len(results)}")
  print(f"Successful experiments: {len(successful_experiments)}")
  print(f"Failed experiments: {len(results) - len(successful_experiments)}")
  print(f"  - Boundary exceeded failures: {len(boundary_failures)}")
  print(f"Success rate: {success_rate:.1f}%")
  
  if successful_experiments:
    avg_iterations = np.mean([r['iterations'] for r in successful_experiments])
    avg_time = np.mean([r['elapsed_time'] for r in successful_experiments])
    avg_path_2d = np.mean([r['path_length_2d'] for r in successful_experiments])
    avg_path_3d = np.mean([r['path_length_3d'] for r in successful_experiments])
    print(f"Average iterations (successful): {avg_iterations:.1f}")
    print(f"Average time (successful): {avg_time:.2f}s")
    print(f"Average 2D path length (successful): {avg_path_2d*1000:.1f} mm")
    print(f"Average 3D path length (successful): {avg_path_3d*1000:.1f} mm")
  
  # ==================== SAVE SUMMARY RESULTS ====================
  summary_data = {
    'primitives': args.primitives,
    'controller': args.controller,
    'channels': args.ch,
    'mode': mode,
    'mode_description': mode_description,
    'total_experiments': len(results),
    'successful_experiments': len(successful_experiments),
    'failed_experiments': len(results) - len(successful_experiments),
    'boundary_exceeded_failures': len(boundary_failures),
    'success_rate': success_rate,
    'results': results
  }
  
  if mode == 'sweep':
    summary_data['yaw_step_deg'] = yaw_step_deg
  
  if successful_experiments:
    summary_data['avg_iterations'] = np.mean([r['iterations'] for r in successful_experiments])
    summary_data['avg_time'] = np.mean([r['elapsed_time'] for r in successful_experiments])
    summary_data['avg_path_length_2d'] = np.mean([r['path_length_2d'] for r in successful_experiments])
    summary_data['avg_path_length_3d'] = np.mean([r['path_length_3d'] for r in successful_experiments])
    summary_data['avg_path_length_2d_mm'] = summary_data['avg_path_length_2d'] * 1000  # Convert to mm for convenience
    summary_data['avg_path_length_3d_mm'] = summary_data['avg_path_length_3d'] * 1000  # Convert to mm for convenience
  
  # ==================== RETURN TO ORIGINAL YAW ====================
  # After completing all experiments, return to original yaw position
  if mode == 'sweep':
    # Calculate total yaw rotated during the sweep
    total_yaw_rotated = cumulative_yaw
    print(f"\n--- Returning to original yaw position ---")
    print(f"Total yaw rotated during sweep: {total_yaw_rotated * 180.0 / np.pi:.1f}°")
    
    # Call the return function
    final_yaw_deg = return_to_original_yaw(rtde_help, original_disEngagePose, total_yaw_rotated, yaw_step_deg)
    
    # Update summary data with return information
    summary_data['return_to_original_yaw'] = {
      'total_yaw_rotated_deg': total_yaw_rotated * 180.0 / np.pi,
      'return_angle_deg': -total_yaw_rotated * 180.0 / (3.0 * np.pi),
      'final_yaw_deg': final_yaw_deg
    }
  else:
    print(f"\n--- Random mode: No return to original yaw needed ---")
    summary_data['return_to_original_yaw'] = None

  # Save summary to file with mode in filename
  summary_filename = f"experiment_summary_{args.primitives}_{args.controller}_ch{args.ch}_{mode}.json"
  summary_path = os.path.join(file_help.ResultSavingDirectory, summary_filename)
  
  with open(summary_path, 'w') as f:
    import json
    json.dump(summary_data, f, indent=2, default=str)
  
  print(f"\nSummary saved to: {summary_path}")
  print("=" * 50)


def main(args):
  """
  Main function to setup and execute haptic search experiments.
  
  This function handles:
    1. Initialization of ROS node and helper objects
    2. Setup of robot TCP offset and sensors
    3. Configuration of experiment parameters
    4. Initialization of RL controller (if specified)
    5. Robot movement to disengage position
    6. Execution of experiments:
       - For trap/dumbbell: Runs 10 repeated experiments with random yaw
       - For other primitives: Runs single experiment with user interaction
  
  The function automatically determines experiment mode based on primitive type:
    - 'trap' or 'dumbbell' → Automated repeated experiments (sweep or random mode)
    - Other primitives → Single manual experiment with user prompts
  
  Args:
    args: Parsed command line arguments containing:
      - primitives: Type of shape ('trap', 'dumbbell', 'polygons')
      - ch: Number of suction cup chambers (3, 4, 5, or 6)
      - controller: Controller type (greedy, yaw_momentum, rl_hgreedy, etc.)
      - max_iterations: Maximum control iterations per experiment
      - reverse: Whether to use reverse airflow
      - yaw_mode: Experiment mode - 'sweep' (default, 72 experiments) or 'random' (10 experiments)
  
  Returns:
    None (experiment data is saved to files)
  """


  deg2rad = np.pi / 180.0
  DUTYCYCLE_100 = 100
  DUTYCYCLE_30 = 30
  DUTYCYCLE_0 = 0

  SYNC_RESET = 0
  SYNC_START = 1
  SYNC_STOP = 2

  np.set_printoptions(precision=4)

  # controller node
  rospy.init_node('suction_cup')

  # Setup helper functions
  FT_help = FT_CallbackHelp() # it deals with subscription.
  rospy.sleep(0.5)
  P_help = P_CallbackHelp() # it deals with subscription.
  rospy.sleep(0.5)
  rtde_help = rtdeHelp(125)
  rospy.sleep(0.5)
  file_help = fileSaveHelp()
  search_help = hapticSearch2DHelp(P_vac = -18000, d_lat = 1.0e-3, d_yaw=1.5, damping_factor=0.7, n_ch = args.ch, p_reverse = args.reverse) # d_lat is important for the haptic search (if it is too small, the controller will fail)

  # Set the TCP offset and calibration matrix
  rospy.sleep(0.5)
  rtde_help.setTCPoffset([0, 0, 0.150, 0, 0, 0])
  # for 5 and 6-chambered suction cups, the 3D printed fixtures are longer than 3 and 4-chambered suction cups.
  if args.ch == 5:
    rtde_help.setTCPoffset([0, 0, 0.150 + 0.019, 0, 0, 0])
  if args.ch == 6:
    rtde_help.setTCPoffset([0, 0, 0.150 + 0.020, 0, 0, 0])
  rospy.sleep(0.2)


  # Set the PWM Publisher  
  targetPWM_Pub = rospy.Publisher('pwm', Int8, queue_size=1)
  rospy.sleep(0.5)
  targetPWM_Pub.publish(DUTYCYCLE_0)

  # Set the synchronization Publisher
  syncPub = rospy.Publisher('sync', Int8, queue_size=1)
  syncPub.publish(SYNC_RESET)

  print("Wait for the data_logger to be enabled")
  rospy.wait_for_service('data_logging')
  dataLoggerEnable = rospy.ServiceProxy('data_logging', Enable)
  dataLoggerEnable(False) # reset Data Logger just in case
  rospy.sleep(1)
  file_help.clearTmpFolder()        # clear the temporary folder
  datadir = file_help.ResultSavingDirectory
  
  # Set the disengage pose
  disengagePosition = get_disEngagePosition(args.primitives)
  # set the default yaw angle in order to have the first chamber of each suction cups facing towards +x-axis when the suction cup is in the disengage pose
  if args.ch == 3:
    default_yaw = pi/2 - 60*pi/180
  if args.ch == 4:
    default_yaw = pi/2 - 45*pi/180
  if args.ch == 5:
    default_yaw = pi/2 - 90*pi/180
  if args.ch == 6:
    default_yaw = pi/2 - 60*pi/180
  setOrientation = tf.transformations.quaternion_from_euler(default_yaw,pi,0,'szxy')
  disEngagePose = rtde_help.getPoseObj(disengagePosition, setOrientation)

  
  # set initial parameters
  suctionSuccessFlag = False
  controller_str = args.controller
  P_vac = search_help.P_vac
  max_steps = args.max_iterations # maximum number of steps to take
  args.max_steps = max_steps
  timeLimit = 15 # maximum time to take (in seconds) - kept for compatibility
  args.timeLimit = timeLimit
  pathLimit = 50e-3 # maximum path length to take (in mm)
  args.pathLimit = pathLimit  
  
  # Initialize RL controller if specified
  rl_controller = None
  if controller_str.startswith('rl_'):
    model_type = controller_str[3:]  # Remove 'rl_' prefix
    try:
      rl_controller = create_rl_controller(args.ch, model_type)
      print(f"RL controller initialized with model: ch{args.ch}_{model_type}")
    except Exception as e:
      print(f"Failed to initialize RL controller: {e}")
      print("Falling back to heuristic controller")
      controller_str = "greedy"  # Fallback to greedy heuristic
  

  # try block so that we can have a keyboard exception
  try:
    input("Press <Enter> to go to disengagepose")
    rtde_help.goToPose(disEngagePose)

    # Check if we should run repeated experiments for trap/dumbbell
    if args.primitives in ['trap', 'dumbbell']:
      if args.multi_controller:
        print(f"Running multi-controller yaw sweep for {args.primitives} primitive")
        multi_controller_results = run_multi_controller_yaw_sweep(args, rtde_help, search_help, P_help, targetPWM_Pub, 
                                                                dataLoggerEnable, file_help, disEngagePose, rl_controller)
        print(f"Multi-controller experiment completed!")
      else:
        print(f"Running repeated experiments for {args.primitives} primitive")
        run_repeated_experiments(args, rtde_help, search_help, P_help, targetPWM_Pub, 
                                dataLoggerEnable, file_help, disEngagePose, rl_controller, 
                                mode=args.yaw_mode)
    else:
      input("Press <Enter> to Start the haptic search with hopping motion")
      
      # Run the haptic search using the common function
      result = run_haptic_search_loop(args, rtde_help, search_help, P_help, targetPWM_Pub, 
                                     dataLoggerEnable, file_help, disEngagePose, rl_controller)

      # save data and clear the temporary folder
      file_help.saveDataParams(args, appendTxt='Jp_2D_HS_hopping_' + 'ch' + str(args.ch) + '_' + str(args.primitives)+'_controller_'+ str(args.controller))
      file_help.clearTmpFolder()
      
      # go to disengage pose
      print("Start to go to disengage pose")
      rtde_help.goToPose(disEngagePose)

    print("============ Python UR_Interface demo complete!")
  except rospy.ROSInterruptException:
    targetPWM_Pub.publish(DUTYCYCLE_0)
    return
  except KeyboardInterrupt:
    targetPWM_Pub.publish(DUTYCYCLE_0)
    return  


if __name__ == '__main__':  
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--primitives', type=str, help='types of primitives (trap, dumbbell, polygons, etc.)', default= "trap")
  parser.add_argument('--ch', type=int, help='number of channel', default= 4)
  parser.add_argument('--controller', type=str, help='2D haptic contollers (greedy, yaw, momentum, yaw_momentum, rl_hgreedy, rl_hmomentum, rl_hyaw, rl_hyaw_momentum). Use --multi_controller to test all four heuristic controllers automatically.', default= "yaw_momentum")
  parser.add_argument('--max_iterations', type=int, help='maximum number of iterations (default: 50)', default= 50)
  parser.add_argument('--reverse', type=bool, help='when we use reverse airflow', default= False)
  parser.add_argument('--yaw_mode', type=str, choices=['sweep', 'random'], 
                      help='yaw experiment mode: sweep (default, 0-360° in steps) or random (10 random angles)', 
                      default='sweep')
  parser.add_argument('--yaw_step', type=float, help='yaw step size in degrees for sweep mode (default: 5°, try 10° or 15° for faster sweeps)', 
                      default=10.0)
  parser.add_argument('--start_experiment', type=int, help='experiment number to start from (default: 1, use this to resume from a specific experiment)', 
                      default=1)
  parser.add_argument('--pause_time', type=float, help='pause time between experiments in seconds (default: 0.5s, try 0.2s for faster)', 
                      default=0.2)
  parser.add_argument('--hop_sleep', type=float, help='sleep time after hopping down in seconds (default: 0.08s, try 0.05s for faster)', 
                      default=0.15)
  parser.add_argument('--multi_controller', action='store_true', 
                      help='run yaw sweep with all four controllers (greedy, yaw, momentum, yaw_momentum) in sequence')

  args = parser.parse_args()    
  
  main(args)
