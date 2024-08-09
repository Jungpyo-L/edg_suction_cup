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

from math import pi, cos, sin


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

  F_normalThres = [1.5, 2.0]

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

  
  # pose initialization
  xoffset = args.xoffset
  disengagePosition_init =  [-0.597, .211, 0.025] # unit is in m
  setOrientation = tf.transformations.quaternion_from_euler(pi,0,pi/2,'sxyz') #static (s) rotating (r)
  disEngagePose = rtde_help.getPoseObj(disengagePosition_init, setOrientation)


  # try block so that we can have a keyboard exception
  try:
    # Go to disengage Pose #test
    input("Press <Enter> to go disEngagePose")
    rtde_help.goToPose(disEngagePose)
    rospy.sleep(0.1)

    input("Press <Enter> to start to data collection")
    for j in range(xoffset, 7):

      print("Move to the upated disengage point")
      # add offset to the disengage position
      args.xoffset = j
      # copy the initial disengage position without changing initial value
      disengagePosition = copy.deepcopy(disengagePosition_init)
      print("disengagePosition: ", disengagePosition)
      disengagePosition[0] += j*0.001
      print("disengagePosition: ", disengagePosition)

      for i in range(round(args.angle/5)+1):

        print("offset: ", j)
        print("Pose Idx: ", i)
        args.theta = round((pi/36*i)*180/pi)
        print("Theta =", args.theta)

        if args.startAngle > args.theta:
          continue
        
        # add yaw angle to the disengage position
        targetOrientation = tf.transformations.quaternion_from_euler(pi,0,pi/2+pi/36*i,'sxyz') #static (s) rotating (r)
        targetPose = rtde_help.getPoseObj(disengagePosition, targetOrientation)
        targetPose_init = targetPose
        rtde_help.goToPose(targetPose)
        # targetPWM_Pub.publish(DUTYCYCLE_100)
        targetPWM_Pub.publish(DUTYCYCLE_0)
        syncPub.publish(SYNC_RESET)
        rospy.sleep(0.1)

        # set pressure and FT offset
        P_help.startSampling()      
        rospy.sleep(0.1) # default is 0.5
        FT_help.setNowAsBias()
        P_help.setNowAsOffset()
        Fz_offset = FT_help.averageFz

        # start data logging
        dataLoggerEnable(True)
        rospy.sleep(0.3) # default is 0.5

        print("move along normal")
        # flags and variables
        farFlag = True
        
        # slow approach until normal force is high enough
        F_normal = FT_help.averageFz_noOffset

        while farFlag:
            if F_normal > -F_normalThres[0]:
              T_move = adpt_help.get_Tmat_TranlateInZ(direction = 1)
              targetPose = adpt_help.get_PoseStamped_from_T_initPose(T_move, targetPose)
              rtde_help.goToPoseAdaptive(targetPose, time = 0.01)
              # rtde_help.goToPose(targetPose)

              # new normal force
              F_normal = FT_help.averageFz_noOffset

            elif F_normal < -F_normalThres[1]:
              T_move = adpt_help.get_Tmat_TranlateInZ(direction = -1)
              targetPose = adpt_help.get_PoseStamped_from_T_initPose(T_move, targetPose)
              rtde_help.goToPoseAdaptive(targetPose, time = 0.01)
              # rtde_help.goToPose(targetPose)
              
              # new normal force
              F_normal = FT_help.averageFz_noOffset

            else:
              farFlag = False
              rtde_help.stopAtCurrPoseAdaptive()
              print("reached threshhold normal force: ", F_normal)
              rospy.sleep(0.1)

        print("Start to record data")
        syncPub.publish(SYNC_START)
        rospy.sleep(1) # default is 2
        P_init = P_help.four_pressure
        P_vac = P_help.P_vac
        if all(np.array(P_init)<P_vac) and i == 0:
          print("Suction Engage Succeed from initial touch")
          SuctionFlag = True
        else:
          SuctionFlag = False
        print("Stop to record data")
        syncPub.publish(SYNC_STOP)
        rospy.sleep(0.1)
        targetPWM_Pub.publish(DUTYCYCLE_0)
        rtde_help.goToPose(targetPose_init)

        # stop data logging
        rospy.sleep(0.2)
        dataLoggerEnable(False)
        rospy.sleep(0.2)    

        # save data and clear the temporary folder
        file_help.saveDataParams(args, appendTxt='jp_lateral_'+'xoffset_' + str(args.xoffset)+'_theta_' + str(args.theta))
        file_help.clearTmpFolder()
        P_help.stopSampling()
        rospy.sleep(0.3) # default is 0.5
        if SuctionFlag == True:
          break

      # in order to go back to the initial orientation
      if SuctionFlag == False:
        for i in range(4):
          targetOrientation = tf.transformations.quaternion_from_euler(pi,0,pi/2-pi/2*(i+1),'sxyz') #static (s) rotating (r)
          targetPose = rtde_help.getPoseObj(disengagePosition, targetOrientation)
          rtde_help.goToPose(targetPose)
          rospy.sleep(0.1)      
    

    print("Go to disengage point")
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
  parser.add_argument('--xoffset', type=float, help='x direction offset', default= 0)
  parser.add_argument('--angle', type=int, help='angles of exploration', default= 360)
  parser.add_argument('--startAngle', type=int, help='angles of exploration', default= 0)
  parser.add_argument('--primitives', type=str, help='types of primitives (edge, corner, etc.)', default= "edge")

  args = parser.parse_args()    
  
  main(args)
  # main(depth, rotAngleList[mode], translateZList[mode])