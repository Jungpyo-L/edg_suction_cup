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
  adpt_help = adaptMotionHelp(dw = 0.5, d_lat = 0.5e-3, d_z = 0.1e-3)

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
  disengagePosition_init =  [-0.581, .206, 0.025] # unit is in m
  setOrientation = tf.transformations.quaternion_from_euler(pi,0,pi/2,'sxyz') #static (s) rotating (r)
  disEngagePose = rtde_help.getPoseObj(disengagePosition_init, setOrientation)


  P_vac = P_help.P_vac
  timeLimit = 15 # sec


  # try block so that we can have a keyboard exception
  try:
    # Go to disengage Pose
    input("Press <Enter> to go disEngagePose")
    rtde_help.goToPose(disEngagePose)
    rospy.sleep(0.1)

    P_help.startSampling()      
    rospy.sleep(0.5)
    FT_help.setNowAsBias()
    P_help.setNowAsOffset()
    Fz_offset = FT_help.averageFz


    input("Press <Enter> to go normal to get engage point")

    if args.zHeight == True:
      with open(datadir + '/engage_z.p', 'rb') as handle:
        engage_z = pickle.load(handle) -4e-3
      
    else:
      initEndEffPoseStamped = rtde_help.getCurrentPose()

      print("move along normal")
      targetPose = rtde_help.getCurrentPose()

      # flags and variables
      
      farFlag = True
      
      # slow approach until normal force is high enough
      F_normal = FT_help.averageFz_noOffset
      targetPWM_Pub.publish(DUTYCYCLE_0)

      while farFlag:
          if F_normal > -F_normalThres[0]:
            T_move = adpt_help.get_Tmat_TranlateInZ(direction = 1)
            targetPose = adpt_help.get_PoseStamped_from_T_initPose(T_move, targetPose)
            rtde_help.goToPoseAdaptive(targetPose, time = 0.1)

            # new normal force
            F_normal = FT_help.averageFz_noOffset

          elif F_normal < -F_normalThres[1]:
            T_move = adpt_help.get_Tmat_TranlateInZ(direction = -1)
            targetPose = adpt_help.get_PoseStamped_from_T_initPose(T_move, targetPose)
            rtde_help.goToPoseAdaptive(targetPose, time = 0.1)
            
            # new normal force
            F_normal = FT_help.averageFz_noOffset

          else:
            farFlag = False
            rtde_help.stopAtCurrPoseAdaptive()
            print("reached threshhold normal force: ", F_normal)
            args.normalForceUsed= F_normal
            rospy.sleep(0.1)
      
      targetPoseEngaged = rtde_help.getCurrentPose()
      T_N_Engaged = adpt_help.get_Tmat_from_Pose(targetPoseEngaged)
      engage_z = targetPoseEngaged.pose.position.z
      rospy.sleep(0.1)


    input("Press <Enter> to start lateral search")
    targetPWM_Pub.publish(DUTYCYCLE_100)
    startTime = time.time()
    # rtde_help = rtdeHelp(125)
    
  
    while True:
            

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

      if F_total > 10:
        print("net force acting on cup is too high")

        # stop at the last pose
        rtde_help.stopAtCurrPose()
        rospy.sleep(0.1)
        sequentialFailures+=1
        targetPWM_Pub.publish(DUTYCYCLE_0)
        break

      # get FT and quat for FT control
      (trans, quat) = rtde_help.readCurrPositionQuat()
      T_array_cup = adpt_help.get_T_array_cup(T_array, F_array, quat)

      # calculate transformation matrices
      T_align, T_later = adpt_help.get_Tmats_from_controller(P_array, T_array_cup, 'W1', 1)
      T_normalMove = adpt_help.get_Tmat_axialMove(F_normal, F_normalThres)
      T_move =  T_later @ T_align @ T_normalMove # lateral --> align --> normal

      # move to new pose adaptively
      measuredCurrPose = rtde_help.getCurrentPose()
      currPose = adpt_help.get_PoseStamped_from_T_initPose(T_move, measuredCurrPose)
      rtde_help.goToPoseAdaptive(currPose)


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

      elif time.time()-startTime >timeLimit:
            args.timeOverFlag = time.time()-startTime >timeLimit

            suctionSuccessFlag = False
            print("Suction controller failed!")

            # stop at the last pose
            rtde_help.stopAtCurrPoseAdaptive()
            targetPWM_Pub.publish(DUTYCYCLE_0)
            
            # keep X sec of data after alignment is complete
            rospy.sleep(0.1)
            break

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


  args = parser.parse_args()    
  
  main(args)
  # main(depth, rotAngleList[mode], translateZList[mode])