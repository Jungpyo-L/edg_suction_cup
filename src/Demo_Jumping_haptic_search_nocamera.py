#!/usr/bin/env python

# Authors: Jungpyo Lee

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


from helperFunction.SuctionP_callback_helper import P_CallbackHelp
from helperFunction.FT_callback_helper import FT_CallbackHelp
from helperFunction.fileSaveHelper import fileSaveHelp
from helperFunction.rtde_helper import rtdeHelp
from helperFunction.adaptiveMotion import adaptMotionHelp


def main(args):

  deg2rad = np.pi / 180.0
  DUTYCYCLE_100 = 100
  DUTYCYCLE_30 = 30
  DUTYCYCLE_0 = 0

  SYNC_RESET = 0
  SYNC_START = 1
  SYNC_STOP = 2
  timeLimit = 15.0 # searching time limit  
  dispLimit = 50e-3 # displacement limit

 

  F_normalThres = [args.normalForce, args.normalForce+0.5]
  args.normalForce_thres = F_normalThres

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
  adpt_help = adaptMotionHelp(dw = 0.5, d_lat = 5e-3, d_z = 0.1e-3)

  # Set the TCP offset and calibration matrix
  rospy.sleep(0.5)
  rtde_help.setTCPoffset([0, 0, 0.150, 0, 0, 0])
  rospy.sleep(0.2)
  rtde_help.setCalibrationMatrix()
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
  
  # pose initialization
  xoffset = args.xoffset
  disengagePosition_init =  [-0.451, .206, 0.025] # unit is in m
  setOrientation = tf.transformations.quaternion_from_euler(pi,0,pi/2,'sxyz') #static (s) rotating (r)
  disEngagePose = rtde_help.getPoseObj(disengagePosition_init, setOrientation)

  EngagePosition_init =  [-0.451, .206, 0.010] # unit is in m
  EngagePose = rtde_help.getPoseObj(EngagePosition_init, setOrientation)


  P_vac = P_help.P_vac
  timeLimit = 15 # sec


  # try block so that we can have a keyboard exception
  try:

    print("Start to approach to target Pose")
    targetPoseStamped = EngagePose
    T_offset = adpt_help.get_Tmat_TranlateInBodyF([0., 0., -15e-3]) # small offset from the target pose
    targetSearchPoseStamped = adpt_help.get_PoseStamped_from_T_initPose(T_offset, targetPoseStamped)
    rtde_help.goToPose(targetSearchPoseStamped)
    rospy.sleep(0.1)

    P_help.startSampling()      
    rospy.sleep(0.5)
    FT_help.setNowAsBias()
    P_help.setNowAsOffset()
    Fz_offset = FT_help.averageFz
    T_offset = adpt_help.get_Tmat_TranlateInBodyF([0., 0., -15e-3]) # small offset from the target pose


    input("press <Enter> to start haptic search")
    # print("Start to haptic search")
    # flag for alternating, time that controller starts trying to grasp
    PFlag = False
    startTime = time.time()
    alternateTime = time.time()
    prevTime = 0
    suctionSuccessFlag = False
    targetPWM_Pub.publish(DUTYCYCLE_100)

    # get initial pose to check limits
    T_N_Engaged = adpt_help.get_Tmat_from_Pose(targetPoseStamped) # relative angle limit
    T_Engaged_N = np.linalg.inv(T_N_Engaged)

    while not suctionSuccessFlag:
      # 1. down to target pose
      rtde_help.goToPose(targetPoseStamped, speed=args.speed, acc=args.acc)
      rospy.sleep(args.waitTime)

      # calculate axis angle
      measuredCurrPose = rtde_help.getCurrentPose()
      T_N_curr = adpt_help.get_Tmat_from_Pose(measuredCurrPose)          
      T_Engaged_curr = T_Engaged_N @ T_N_curr

      # calculate current pos/angle
      displacement = np.linalg.norm(T_Engaged_curr[0:3,3])

      # 2. get sensor data
      # PFT arrays to calculate Transformation matrices
      P_array = P_help.four_pressure
      T_array = [FT_help.averageTx_noOffset, FT_help.averageTy_noOffset]
      F_array = [FT_help.averageFx_noOffset, FT_help.averageFy_noOffset]
      F_normal = FT_help.averageFz_noOffset

      # check force limits
      Fx = F_array[0]
      Fy = F_array[1]
      Fz = F_normal
      F_total = np.sqrt(Fx**2 + Fy**2 + Fz**2)

      if F_total > 12:
        print("net force acting on cup is too high")
        args.forceLimitFlag = True
        # stop at the last pose
        rtde_help.stopAtCurrPose()
        rospy.sleep(0.1)
        break
      
      # 3. lift up and check suction success
      targetSearchPoseStamped = adpt_help.get_PoseStamped_from_T_initPose(T_offset, targetPoseStamped)
      rtde_help.goToPose(targetSearchPoseStamped, speed=args.speed, acc=args.acc)
      
      # P_vac = P_help.P_vac
      P_vac = -2500.0   
      P_array_check = P_help.four_pressure
      reached_vacuum = all(np.array(P_array_check)<P_vac)

      if reached_vacuum:
        # vacuum seal formed, success!
        suctionSuccessFlag = True
        print("Suction engage succeeded with controller")

        # stop at the last pose
        rtde_help.stopAtCurrPoseAdaptive()
        args.elapedTime = time.time()-startTime
        break

      elif time.time()-startTime > timeLimit or displacement > dispLimit:
        args.timeOverFlag = time.time()-startTime >timeLimit
        args.dispOverFlag = displacement > dispLimit

        suctionSuccessFlag = False
        print("Haptic search fail")
        # stop at the last pose
        rtde_help.stopAtCurrPoseAdaptive()
        targetPWM_Pub.publish(DUTYCYCLE_0)
        rospy.sleep(0.1)
        break

      # 4. move to above the next search location
      T_later = adpt_help.get_Tmat_lateralMove(P_array)
      targetPoseStamped = adpt_help.get_PoseStamped_from_T_initPose(T_later, measuredCurrPose)
      targetSearchPoseStamped = adpt_help.get_PoseStamped_from_T_initPose(T_offset, targetPoseStamped)
      rtde_help.goToPose(targetSearchPoseStamped, speed=args.speed, acc=args.acc)

    input("press enter to go disengage pose")
    rospy.sleep(0.1)
    targetPWM_Pub.publish(DUTYCYCLE_0)
    setOrientation = tf.transformations.quaternion_from_euler(pi,0,pi/2,'sxyz') #static (s) rotating (r)
    disEngagePose = rtde_help.getPoseObj(disengagePosition_init, setOrientation)
    rtde_help.goToPose(disEngagePose)
    # cartesian_help.goToPose(disEngagePose,wait=True)
    rospy.sleep(0.3)

    print("============ Stopping data logger ...")
    print("before dataLoggerEnable(False)")
    print(dataLoggerEnable(False)) # Stop Data Logging
    print("after dataLoggerEnable(False)")
  

    P_help.stopSampling()


    print("============ Python UR_Interface demo complete!")
  
  except rospy.ROSInterruptException:
    return
  except KeyboardInterrupt:
    return  


if __name__ == '__main__':  
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--useStoredData', type=bool, help='take image or use existingFile', default=False)
  parser.add_argument('--storedDataDirectory', type=str, help='location of target saved File', default="")
  parser.add_argument('--startIdx', type=int, help='startIndex Of the pose List', default= 0)
  parser.add_argument('--xoffset', type=int, help='x direction offset', default= 0)
  parser.add_argument('--angle', type=int, help='angles of exploration', default= 360)
  parser.add_argument('--startAngle', type=int, help='angles of exploration', default= 0)
  parser.add_argument('--primitives', type=str, help='types of primitives (edge, corner, etc.)', default= "edge")
  parser.add_argument('--normalForce', type=float, help='normal force', default= 1.5)
  parser.add_argument('--zHeight', type=bool, help='use presset height mode? (rather than normal force)', default= False)
  parser.add_argument('--ch', type=int, help='number of channel', default= 4)
  parser.add_argument('--speed', type=float, help='take image or use existingFile', default=0.3)
  parser.add_argument('--acc', type=float, help='location of target saved File', default=0.6)
  parser.add_argument('--waitTime', type=float, help='location of target saved File', default=0.1)
  parser.add_argument('--dLateral', type=float, help='location of target saved File', default=0.005)


  args = parser.parse_args()    
  
  main(args)
  # main(depth, rotAngleList[mode], translateZList[mode])