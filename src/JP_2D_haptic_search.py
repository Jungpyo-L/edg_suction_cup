#!/usr/bin/env python

# Authors: Jungpyo Lee
# Create: Aug.18.2024
# Last update: Aug.18.2024
# Description: This script is used to test 2D haptic search models while recording pressure and path.

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


def main(args):
  if args.ch == 6:
    from helperFunction.SuctionP_callback_helper_ch6 import P_CallbackHelp
  else:
    from helperFunction.SuctionP_callback_helper import P_CallbackHelp

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
  rtde_help = rtde_help = rtdeHelp(125)
  rospy.sleep(0.5)
  file_help = fileSaveHelp()
  search_help = hapticSearch2DHelp(dw = 0.5, d_lat = 0.5e-3, d_z = 0.05e-3, d_yaw=1, n_ch = args.ch)

  # Set the TCP offset and calibration matrix
  rospy.sleep(0.5)
  rtde_help.setTCPoffset([0, 0, 0.150, 0, 0, 0])
  if args.ch == 6:
    rtde_help.setTCPoffset([0, 0, 0.150 + 0.02, 0, 0, 0])
  rospy.sleep(0.2)

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
  
  # Set the disengage pose
  disengagePosition =  [-0.630, .275, 0.0157]
  setOrientation = tf.transformations.quaternion_from_euler(pi,0,pi/2 + pi/180*args.yaw,'sxyz') #static (s) rotating (r)
  disEngagePose = rtde_help.getPoseObj(disengagePosition, setOrientation)
  
  # set initial parameters
  suctionSuccessFlag = False
  controller_str = args.controller
  
  # try block so that we can have a keyboard exception
  try:
    input("Press <Enter> to go to disengagepose")
    rtde_help.goToPose(disEngagePose)
    
    input("Press <Enter> to start the recording")
    P_help.startSampling()      
    rospy.sleep(0.5)
    P_help.setNowAsOffset()
    dataLoggerEnable(True)

    input("Press <Enter> to go to engagepose")
    engagePosition = copy.deepcopy(disengagePosition)
    engagePosition[2] = disengagePosition[2] - 5e03
    engagePose = rtde_help.getPoseObj(engagePosition, setOrientation)
    rtde_help.goToPose(engagePose)

    input("Press <Enter> to start the haptic search")
    while not suctionSuccessFlag:   # while no success in grasp, run controller until success or timeout
      
      # PFT arrays to calculate Transformation matrices
      P_array = P_help.four_pressure
      
      # get the current yaw angle of the suction cup
      # Todo : get the yaw angle from the controller
      measuredCurrPose = rtde_help.getCurrentPose()

      # calculate transformation matrices
      T_align, T_later = search_help.get_Tmats_from_controller(P_array, yaw_angle, controller_str)
      T_normalMove = search_help.get_Tmat_axialMove(F_normal, F_normalThres)
      T_move =  T_later @ T_align @ T_normalMove # lateral --> align --> normal

      # move to new pose adaptively
      measuredCurrPose = rtde_help.getCurrentPose()
      currPose = search_help.get_PoseStamped_from_T_initPose(T_move, measuredCurrPose)
      rtde_help.goToPoseAdaptive(currPose)

      # calculate axis angle
      T_N_curr = search_help.get_Tmat_from_Pose(measuredCurrPose)          
      T_Engaged_curr = T_Engaged_N @ T_N_curr
      currAxisAngleToZ = np.arccos(T_N_curr[2,2])

      # calculate current pos/angle
      displacement = np.linalg.norm(T_Engaged_curr[0:3,3])
      angleDiff = np.arccos(T_Engaged_curr[2,2])

      #=================== check attempt break conditions =================== 

      # LOOP BREAK CONDITION 1
      reached_vacuum = all(np.array(P_array)<P_vac)
      # print("loop time: ", time.time()-prevTime)
      # if time.time()-prevTime > 0.1:
        # print("Time exceed 0.1 sec~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
      # print("np.array(P_array): ", np.array(P_array))
      # print("reached vacuum: ", reached_vacuum)

      if reached_vacuum:
        # vacuum seal formed, success!
        suctionSuccessFlag = True
        print("Suction engage succeeded with controller")

        # stop at the last pose
        rtde_help.stopAtCurrPoseAdaptive()

        # keep X sec of data after alignment is complete
        rospy.sleep(0.1)
        break
      
      # LOOP BREAK CONDITION 2
      # if timeout, or displacement/angle passed, failed
      elif time.time()-startTime >timeLimit or displacement > dispLimit or angleDiff > angleLimit or currAxisAngleToZ < (np.pi*2/3):
        args.timeOverFlag = time.time()-startTime >timeLimit
        args.dispOverFlag = displacement > dispLimit
        args.angleOverFlag = angleDiff > angleLimit
        args.angleTooFlatFlag = currAxisAngleToZ < (np.pi*2/3)

        suctionSuccessFlag = False
        print("Suction controller failed!")
        sequentialFailures+=1

        # stop at the last pose
        rtde_help.stopAtCurrPoseAdaptive()
        targetPWM_Pub.publish(DUTYCYCLE_0)
        

        # keep X sec of data after alignment is complete
        rospy.sleep(0.1)
        break





    input("Press <Enter> to go stop the recording")
    # stop data logging
    rospy.sleep(0.2)
    dataLoggerEnable(False)
    rospy.sleep(0.2)
    P_help.stopSampling()

    # save data and clear the temporary folder
    file_help.saveDataParams(args, appendTxt='jp_lateral_'+'corner_' + str(args.corner)+'_xoffset_' + str(args.xoffset)+'_theta_' + str(args.theta)+'_material_' + str(args.material))
    file_help.clearTmpFolder()
    


    print("============ Python UR_Interface demo complete!")
  except rospy.ROSInterruptException:
    return
  except KeyboardInterrupt:
    return  


if __name__ == '__main__':  
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--primitives', type=str, help='types of primitives (edge, corner, etc.)', default= "edge")
  parser.add_argument('--ch', type=int, help='number of channel', default= 4)
  parser.add_argument('--controller', type=str, help='2D haptic contollers', default= "normal")
  parser.add_argument('--material', type=int, help='Moldmax: 0, Elastic50: 1, agilus30: 2', default= 1)
  parser.add_argument('--tilt', type=int, help='tilted angle of the suction cup', default= 0)
  parser.add_argument('--yaw', type=int, help='yaw angle of the suction cup', default= 0)
  parser.add_argument('--suction', type=bool, help='whether suction succeed or not', default= False)


  args = parser.parse_args()    
  
  main(args)
  # main(depth, rotAngleList[mode], translateZList[mode])