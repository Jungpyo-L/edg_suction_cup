#!/usr/bin/env python

# Authors: Jungpyo Lee
# Create: Oct.08.2025
# Last update: Oct.08.2025
# Description: This script will test haptic search controllers in the air environment.

# imports
try:
  import rospy
  import tf
  ros_enabled = True
except:
  print('Couldn\'t import ROS.  I assume you\'re running this on your laptop')
  ros_enabled = False

import os, sys
import numpy as np
import time
import scipy
from scipy import signal
from math import pi, cos, sin, floor

from netft_utils.srv import *
from suction_cup.srv import *
from std_msgs.msg import String
from std_msgs.msg import Int8
import geometry_msgs.msg
import tf

from helperFunction.FT_callback_helper import FT_CallbackHelp
from helperFunction.fileSaveHelper import fileSaveHelp
from helperFunction.rtde_helper import rtdeHelp
from helperFunction.hapticSearch2D import hapticSearch2DHelp
from helperFunction.SuctionP_callback_helper import P_CallbackHelp
from helperFunction.RL_controller_helper import RLControllerHelper, create_rl_controller
from helperFunction.utils import hat, create_transform_matrix


def convert_yawAngle(yaw_radian):
  yaw_angle = yaw_radian*180/pi
  yaw_angle = -yaw_angle - 90
  return yaw_angle


def main(args):


  deg2rad = np.pi / 180.0
  DUTYCYCLE_100 = 100
  DUTYCYCLE_30 = 30
  DUTYCYCLE_0 = 0

  SYNC_RESET = 0
  SYNC_START = 1
  SYNC_STOP = 2

  FT_SimulatorOn = False
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

  if FT_SimulatorOn:
    print("wait for FT simul")
    rospy.wait_for_service('start_sim')
    # bring the service
    netftSimCall = rospy.ServiceProxy('start_sim', StartSim)

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
  disEngagePosition =  [-0.642, .290, 0.4]
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
  disEngagePose = rtde_help.getPoseObj(disEngagePosition, setOrientation)

  
  # set initial parameters
  suctionSuccessFlag = False
  controller_str = getattr(args, 'controller', 'greedy')
  P_vac = search_help.P_vac
  max_steps = 50 # maximum number of steps to take
  args.max_steps = max_steps
  timeLimit = 15 # maximum time to take (in seconds)
  args.timeLimit = timeLimit
  distanceLimit = 30e-3 # maximum distance from the initial position to take (in mm)
  args.distanceLimit = distanceLimit  
  
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
    
    print("Start the haptic search")
    targetPWM_Pub.publish(DUTYCYCLE_100)
    P_help.startSampling()      
    rospy.sleep(1)
    P_help.setNowAsOffset()
    dataLoggerEnable(True)

    # set initial parameters
    print(f"Start the haptic search with controller: {controller_str}")
    suctionSuccessFlag = False
    P_vac = search_help.P_vac
    startTime = time.time()
    iteration_count = 0 # to check the frequency of the loop
    
    # Reset RL controller history if using RL
    if rl_controller and rl_controller.is_model_loaded():
      rl_controller.reset_history()
      print("RL controller history reset")
    
    # begin the haptic search while continuous motion until timeout
    while not time.time()-startTime >timeLimit:   #run controller until timeout
      P_array = P_help.four_pressure      
      
      # get the current yaw angle of the suction cup
      measuredCurrPose = rtde_help.getCurrentPose()
      T_curr = search_help.get_Tmat_from_Pose(measuredCurrPose)
      yaw_angle = convert_yawAngle(search_help.get_yawRotation_from_T(T_curr))

      # calculate transformation matrices
      if rl_controller and rl_controller.is_model_loaded():
        # Use RL controller
        try:
          lateral_vel, yaw_vel, debug_info = rl_controller.compute_action(P_array, yaw_angle, return_debug=True)
          
          # Convert RL output to transformation matrices
          # RL controller outputs velocity, we need to convert to step size
          step_lateral = search_help.d_lat
          step_yaw = search_help.d_yaw
          
          # Create lateral movement transformation
          dx_lat = lateral_vel[0] * step_lateral
          dy_lat = lateral_vel[1] * step_lateral
          T_later = search_help.get_Tmat_TranlateInBodyF([dx_lat, dy_lat, 0.0])
          
          # Create yaw rotation transformation
          if abs(yaw_vel) > 1e-6:  # Only rotate if there's significant yaw velocity
            d_yaw_rad = yaw_vel * step_yaw * np.pi / 180.0
            rot_axis = np.array([0, 0, -1])
            omega_hat = hat(rot_axis)
            Rw = scipy.linalg.expm(d_yaw_rad * omega_hat)
            T_yaw = create_transform_matrix(Rw, [0, 0, 0])
          else:
            T_yaw = np.eye(4)
          
          T_align = np.eye(4)
          
          # Print debug info occasionally
          if iteration_count % 50 == 0:
            print(f"RL Controller - Lateral: {lateral_vel}, Yaw: {yaw_vel:.3f}")
            print(f"Debug: {debug_info}")
            
        except Exception as e:
          print(f"RL controller failed: {e}, falling back to heuristic")
          T_later, T_yaw, T_align = search_help.get_Tmats_from_controller(P_array, "greedy")
      else:
        # Use original heuristic controller
        T_later, T_yaw, T_align = search_help.get_Tmats_from_controller(P_array, controller_str)
      
      T_move =  T_later @ T_yaw @ T_align  # lateral --> align --> normal

      # move to new pose adaptively with continuous motion
      measuredCurrPose = rtde_help.getCurrentPose()
      currPose = search_help.get_PoseStamped_from_T_initPose(T_move, measuredCurrPose)
      rtde_help.goToPose_2Dhaptic(currPose)
      rospy.sleep(0.05)
      
      # calculate current angle position
      measuredCurrPose = rtde_help.getCurrentPose()
      T_curr = search_help.get_Tmat_from_Pose(measuredCurrPose)
      args.final_yaw = convert_yawAngle(search_help.get_yawRotation_from_T(T_curr))


      #=================== check attempt break conditions =================== 
      # Note: Testing mode - no success condition check, runs for full time limit
      # The while loop will continue until time limit is exceeded

      # check loop frequency
      iteration_count += 1

      # Measure frequency every 100 iterations
      if iteration_count % 100 == 0:
          current_time = time.time()
          elapsed_time = current_time - startTime
          frequency = iteration_count / elapsed_time
          # print(f"Current control frequency after {iteration_count} iterations: {frequency} Hz")
      

    # Testing mode - loop completed after time limit
    args.timeOverFlag = True  # Always true since we run for full time limit
    args.elapsedTime = time.time()-startTime
    args.suction = False  # Always false in testing mode
    args.iteration_count = iteration_count
    
    print(f"Controller testing completed after {args.elapsedTime:.2f} seconds with {iteration_count} iterations")
    # stop at the last pose
    rtde_help.stopAtCurrPoseAdaptive()
    targetPWM_Pub.publish(DUTYCYCLE_0)
    # keep X sec of data after testing is complete
    rospy.sleep(0.1)
    print("Press <Enter> to go stop the recording")
    # stop data logging
    rospy.sleep(0.2)
    dataLoggerEnable(False)
    rospy.sleep(0.2)
    P_help.stopSampling()
    targetPWM_Pub.publish(DUTYCYCLE_0)

    # save data and clear the temporary folder
    file_help.saveDataParams(args, appendTxt='jp_2D_HapticSearch_test_controller_'+ str(args.controller))
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
  parser.add_argument('--ch', type=int, help='number of channel', default= 4)
  parser.add_argument('--controller', type=str, help='2D haptic contollers (greedy, yaw, momentum, yaw_momentum, rl_hgreedy, rl_hmomentum, rl_hyaw, rl_hyaw_momentum)', default= "greedy")
  parser.add_argument('--reverse', type=bool, help='when we use reverse airflow', default= False)

  args = parser.parse_args()    
  
  main(args)
