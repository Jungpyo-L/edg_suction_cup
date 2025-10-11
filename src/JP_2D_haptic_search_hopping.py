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
    disEngagePosition =  [0.443, -.038, 0.013]
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
  # Calculate engage position: 5mm below the disengage position
  # This is the starting height for haptic search with contact to the object
  engagePosition = copy.deepcopy(disEngagePose.pose.position)
  engagePosition[2] = engagePosition[2] - 5e-3  # Lower by 5mm (critical for proper seal formation)
  engagePose = rtde_help.getPoseObj(engagePosition, disEngagePose.pose.orientation)
  rtde_help.goToPose_2Dhaptic(engagePose)
  
  # ==================== INITIALIZE CONTROL LOOP VARIABLES ====================
  suctionSuccessFlag = False  # Becomes True when vacuum seal is achieved
  startTime = time.time()     # For computing elapsed time
  iteration_count = 1         # Control iteration counter (starts at 1 for first check)
  pose_diff = 0               # Tracks Z-position error for compensation
  
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
    
    # HOP UP: Lift 5mm to disengage from surface
    currPose.pose.position.z = currPose.pose.position.z + 5e-3
    rtde_help.goToPose_2Dhaptic(currPose)
    rospy.sleep(0.05)  # Brief pause at top of hop
    
    # HOP DOWN: Return to contact, compensating for any Z-position error
    currPose.pose.position.z = currPose.pose.position.z - 5e-3 - pose_diff
    rtde_help.goToPose_2Dhaptic(currPose)
    rospy.sleep(0.05)  # Brief pause at bottom of hop
    
    # ===== UPDATE STATE =====
    # Measure actual position and update error compensation
    measuredCurrPose = rtde_help.getCurrentPose()
    T_curr = search_help.get_Tmat_from_Pose(measuredCurrPose)
    args.final_yaw = convert_yawAngle(search_help.get_yawRotation_from_T(T_curr))
    pose_diff = measuredCurrPose.pose.position.z - currPose.pose.position.z  # Z-error for next iteration
    
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
  
  return {
    'success': suctionSuccessFlag,
    'iterations': iteration_count,
    'elapsed_time': args.elapsedTime if hasattr(args, 'elapsedTime') else 0,
    'final_yaw': args.final_yaw if hasattr(args, 'final_yaw') else 0,
    'path_length_2d': path_length_2d,
    'path_length_3d': path_length_3d
  }

def run_single_experiment(args, rtde_help, search_help, P_help, targetPWM_Pub, 
                         dataLoggerEnable, file_help, disEngagePose, rl_controller, 
                         experiment_num, random_yaw):
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
  
  # Add random yaw to the existing yaw angle
  new_yaw = current_euler[0] + random_yaw
  setOrientation = tf.transformations.quaternion_from_euler(new_yaw, current_euler[1], current_euler[2], 'szxy')
  
  # Create a new disEngagePose with the random yaw for this experiment
  disEngagePose_randomYaw = copy.deepcopy(disEngagePose)
  disEngagePose_randomYaw.pose.orientation.x = setOrientation[0]
  disEngagePose_randomYaw.pose.orientation.y = setOrientation[1]
  disEngagePose_randomYaw.pose.orientation.z = setOrientation[2]
  disEngagePose_randomYaw.pose.orientation.w = setOrientation[3]
  
  print(f"Experiment {experiment_num}: Random yaw = {random_yaw*180/np.pi:.1f} degrees")
  
  # Run the haptic search using the common function
  result = run_haptic_search_loop(args, rtde_help, search_help, P_help, targetPWM_Pub, 
                                 dataLoggerEnable, file_help, disEngagePose_randomYaw, rl_controller, 
                                 experiment_num, random_yaw)

  # Save data for this experiment
  args.experiment_num = experiment_num
  args.random_yaw = random_yaw
  file_help.saveDataParams(args, appendTxt=f'jp_2D_HapticSearch_hopping_{args.primitives}_controller_{args.controller}_exp{experiment_num:02d}_yaw{random_yaw*180/np.pi:.1f}deg')
  file_help.clearTmpFolder()
  
  # Return to disengage pose
  rtde_help.goToPose(disEngagePose)
  
  return {
    'experiment_num': experiment_num,
    'success': result['success'],
    'iterations': result['iterations'],
    'elapsed_time': result['elapsed_time'],
    'random_yaw': random_yaw,
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
  if mode == 'sweep':
    # YAW SWEEP MODE: Systematically sweep through 360 degrees
    yaw_step_deg = 5  # Step size in degrees
    num_experiments = int(360 / yaw_step_deg)  # 72 experiments for 5-degree steps
    yaw_angles = [i * yaw_step_deg * np.pi / 180.0 for i in range(num_experiments)]  # Convert to radians
    mode_description = f"yaw sweep (0° to 355° in {yaw_step_deg}° steps)"
  else:
    # RANDOM YAW MODE: Random yaw angles for each experiment
    num_experiments = 10
    yaw_angles = [generate_random_yaw() for _ in range(num_experiments)]
    mode_description = "random yaw angles"
  
  results = []
  
  print(f"Starting {num_experiments} repeated experiments for {args.primitives}")
  print(f"Mode: {mode_description}")
  print("=" * 50)
  
  # ==================== RUN EXPERIMENTS ====================
  for i in range(num_experiments):
    yaw_angle = yaw_angles[i]
    yaw_deg = yaw_angle * 180.0 / np.pi
    
    print(f"\n--- Experiment {i+1}/{num_experiments} (Yaw: {yaw_deg:.1f}°) ---")
    
    # Run single experiment
    result = run_single_experiment(
      args, rtde_help, search_help, P_help, targetPWM_Pub,
      dataLoggerEnable, file_help, disEngagePose, rl_controller,
      i+1, yaw_angle
    )
    
    results.append(result)
    
    # Brief pause between experiments
    if i < num_experiments - 1:
      print("Waiting 2 seconds before next experiment...")
      rospy.sleep(2)
  
  # Print summary statistics
  print("\n" + "=" * 50)
  print("EXPERIMENT SUMMARY")
  print("=" * 50)
  
  successful_experiments = [r for r in results if r['success']]
  success_rate = len(successful_experiments) / len(results) * 100
  
  print(f"Total experiments: {len(results)}")
  print(f"Successful experiments: {len(successful_experiments)}")
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
  search_help = hapticSearch2DHelp(d_lat = 0.5e-3, d_yaw=1.5, n_ch = args.ch, p_reverse = args.reverse) # d_lat is important for the haptic search (if it is too small, the controller will fail)

  # Set the TCP offset and calibration matrix
  rospy.sleep(0.5)
  rtde_help.setTCPoffset([0, 0, 0.150, 0, 0, 0])
  # for 5 and 6-chambered suction cups, the 3D printed fixtures are longer than 3 and 4-chambered suction cups.
  if args.ch == 6 or args.ch == 5:
    rtde_help.setTCPoffset([0, 0, 0.150 + 0.02, 0, 0, 0])
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
      file_help.saveDataParams(args, appendTxt='jp_2D_HapticSearch_hopping_' + str(args.primitives)+'_controller_'+ str(args.controller))
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
  parser.add_argument('--controller', type=str, help='2D haptic contollers (greedy, yaw, momentum, yaw_momentum, rl_hgreedy, rl_hmomentum, rl_hyaw, rl_hyaw_momentum)', default= "yaw_momentum")
  parser.add_argument('--max_iterations', type=int, help='maximum number of iterations (default: 100)', default= 100)
  parser.add_argument('--reverse', type=bool, help='when we use reverse airflow', default= False)
  parser.add_argument('--yaw_mode', type=str, choices=['sweep', 'random'], 
                      help='yaw experiment mode: sweep (default, 0-360° in 5° steps) or random (10 random angles)', 
                      default='sweep')

  args = parser.parse_args()    
  
  main(args)
