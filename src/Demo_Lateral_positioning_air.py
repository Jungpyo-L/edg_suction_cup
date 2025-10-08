#!/usr/bin/env python

# Authors: Jungpyo Lee
# Description: Simple lateral positioning controller test without force checking or vacuum success detection

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

from helperFunction.utils import rotation_from_quaternion, create_transform_matrix, quaternion_from_matrix, normalize, hat


from helperFunction.SuctionP_callback_helper import P_CallbackHelp
from helperFunction.fileSaveHelper import fileSaveHelp
from helperFunction.rtde_helper import rtdeHelp
from helperFunction.adaptiveMotion import adaptMotionHelp
from helperFunction.hapticSearch2D import hapticSearch2DHelp


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
  P_help = P_CallbackHelp() # it deals with subscription.
  rospy.sleep(0.5)
  rtde_help = rtdeHelp(125)
  rospy.sleep(0.5)
  file_help = fileSaveHelp()
  adpt_help = adaptMotionHelp(dw = 0.5, d_lat = 2e-3, d_z = 0.1e-3)
  search_help = hapticSearch2DHelp(d_lat = 5e-3, d_yaw=1.5, n_ch = args.ch, p_reverse = args.reverse) # d_lat is important for the haptic search (if it is too small, the controller will fail)

  # Set the TCP offset and calibration matrix
  # rospy.sleep(0.5)
  # rtde_help.setTCPoffset([0, 0, 0.150, 0, 0, 0])


  if FT_SimulatorOn:
    print("wait for FT simul")
    rospy.wait_for_service('start_sim')
    # bring the service
    netftSimCall = rospy.ServiceProxy('start_sim', StartSim)

  # Set the PWM Publisher  
  targetPWM_Pub = rospy.Publisher('pwm', Int8, queue_size=1)
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
  
  # pose initialization
  disengagePosition_init =  [0.581, -.206, 0.425] # unit is in m
  if args.ch == 3:
    default_yaw = pi/2 - 60*pi/180
  if args.ch == 4:
    default_yaw = pi/2 - 45*pi/180
  if args.ch == 5:
    default_yaw = pi/2 - 90*pi/180
  if args.ch == 6:
    default_yaw = pi/2 - 60*pi/180
  setOrientation = tf.transformations.quaternion_from_euler(default_yaw,pi,0,'szxy')
  disEngagePose = rtde_help.getPoseObj(disengagePosition_init, setOrientation)


  P_vac = P_help.P_vac
  timeLimit = 10 # sec


  # try block so that we can have a keyboard exception
  try:
    # Go to disengage Pose
    input("Press <Enter> to go disEngagePose")
    rtde_help.goToPose(disEngagePose)
    rospy.sleep(0.1)

    print("Start sampling")
    P_help.startSampling()      
    rospy.sleep(0.5)
    P_help.setNowAsOffset()



    input("Press <Enter> to start lateral search")
    targetPWM_Pub.publish(DUTYCYCLE_100)
    dataLoggerEnable(True)  # Start data logging
    startTime = time.time()
    
  
    while not time.time()-startTime > timeLimit:  # Run until timeout

      # Get pressure array for controller
      P_array = P_help.four_pressure

      # calculate transformation matrices
      T_later, T_yaw, T_align = search_help.get_Tmats_from_controller(P_array, "greedy")
      T_move = T_later @ T_yaw @ T_align # lateral --> yaw --> align

      # move to new pose adaptively
      measuredCurrPose = rtde_help.getCurrentPose()
      currPose = search_help.get_PoseStamped_from_T_initPose(T_move, measuredCurrPose)
      rtde_help.goToPoseAdaptive(currPose)
      
      # Small delay for control loop
      rospy.sleep(0.05)

    # Controller testing completed after timeout
    args.timeOverFlag = True
    args.elapsedTime = time.time()-startTime
    print(f"Controller testing completed after {args.elapsedTime:.2f} seconds")
    
    # stop at the last pose
    rtde_help.stopAtCurrPoseAdaptive()
    targetPWM_Pub.publish(DUTYCYCLE_0)
    # keep X sec of data after testing is complete
    rospy.sleep(0.1)

    print("============ Stopping data logger ...")
    print("before dataLoggerEnable(False)")
    print(dataLoggerEnable(False)) # Stop Data Logging
    P_help.stopSampling()
    rospy.sleep(0.3)
    print("after dataLoggerEnable(False)")
    
    # Save data
    file_help.saveDataParams(args, appendTxt='demo_lateral_positioning_controller_'+ str(getattr(args, 'controller', 'greedy')))
    file_help.clearTmpFolder()
  

    print("============ Python UR_Interface demo complete!")
  
  except rospy.ROSInterruptException:
    return
  except KeyboardInterrupt:
    return  


if __name__ == '__main__':  
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--ch', type=int, help='number of channel', default= 4)
  parser.add_argument('--reverse', type=bool, help='when we use reverse airflow', default= False)
  parser.add_argument('--controller', type=str, help='2D haptic contollers (greedy, yaw, momentum, yaw_momentum, rl_hgreedy, rl_hmomentum, rl_hyaw, rl_hyaw_momentum)', default= "greedy")


  args = parser.parse_args()    
  
  main(args)
  # main(depth, rotAngleList[mode], translateZList[mode])