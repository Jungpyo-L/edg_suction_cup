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

from helperFunction.FT_callback_helper import FT_CallbackHelp
from helperFunction.fileSaveHelper import fileSaveHelp
from helperFunction.rtde_helper import rtdeHelp
from helperFunction.adaptiveMotion import adaptMotionHelp


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

  F_normalThres = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10]
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
  adpt_help = adaptMotionHelp(dw = 0.5, d_lat = 0.5e-3, d_z = 0.05e-3)

  # Set the TCP offset and calibration matrix
  rospy.sleep(0.5)
  rtde_help.setTCPoffset([0, 0, 0.150, 0, 0, 0])
  if args.ch == 6:
    rtde_help.setTCPoffset([0, 0, 0.150 + 0.02, 0, 0, 0])
  rospy.sleep(0.2)
  # rtde_help.setCalibrationMatrix()
  # rospy.sleep(0.2)

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
  disengagePosition_init =  [-0.623, .28, 0.025] # unit is in m
  if args.ch == 6:
    disengagePosition_init =  [-0.63, .28, 0.025 + 0.02] # unit is in m
  setOrientation = tf.transformations.quaternion_from_euler(pi,0,pi/2,'sxyz') #static (s) rotating (r)
  disEngagePose = rtde_help.getPoseObj(disengagePosition_init, setOrientation)
  targetPWM_Pub.publish(DUTYCYCLE_0)

  # try block so that we can have a keyboard exception
  try:
    # Go to disengage Pose
    input("Press <Enter> to go disEngagePose")
    rtde_help.goToPose(disEngagePose)
    rospy.sleep(0.1)

    
    # change yaw angle
    for yaw in range(0, 360//args.ch, 360//args.ch//3):
      # if yaw == 0:
      #   continue
      args.yaw = yaw
      # args.yaw = 90
      # change tile angle
      for tilt in range(0, 31, 5):
        args.tilt = tilt
        print("tilt: ", tilt)
        setOrientation = tf.transformations.quaternion_from_euler(pi+args.tilt*pi/180,0,pi/2,'sxyz') #static (s) rotating (r)
        if args.yaw != 0:
          setOrientation = tf.transformations.quaternion_from_euler(-pi/2 - pi/180 * args.yaw, pi, args.tilt*pi/180,'szxy') #static (s) rotating (r)

        disEngagePose = rtde_help.getPoseObj(disengagePosition_init, setOrientation)
        rtde_help.goToPose(disEngagePose)
        rospy.sleep(0.3)
        P_help.startSampling()      
        rospy.sleep(0.5)
        FT_help.setNowAsBias()
        P_help.setNowAsOffset()
        Fz_offset = FT_help.averageFz
        
        for normal_thresh in F_normalThres:
          args.normalForce = normal_thresh

          print("Start to go normal to get engage point")
          print("normal_thresh: ", normal_thresh)
          dataLoggerEnable(True)
          rospy.sleep(0.3) # default is 0.5

          print("move along normal")
          targetPose = rtde_help.getCurrentPose()

          # flags and variables
          farFlag = True
          args.SuctionFlag = False
          
          # slow approach until it reach suction engage
          F_normal = FT_help.averageFz_noOffset
          targetPoseEngaged = rtde_help.getCurrentPose()
          # targetPWM_Pub.publish(DUTYCYCLE_0)
          syncPub.publish(SYNC_START)
          while farFlag:
            if F_normal > -args.normalForce:
              T_move = adpt_help.get_Tmat_TranlateInZ(direction = 1)
              targetPose = adpt_help.get_PoseStamped_from_T_initPose(T_move, targetPose)
              rtde_help.goToPoseAdaptive(targetPose, time = 0.1)
              F_normal = FT_help.averageFz_noOffset
              args.normalForceActual = F_normal
              rospy.sleep(0.1)

            else:
              farFlag = False
              rtde_help.stopAtCurrPoseAdaptive()
              print("reached threshhold normal force: ", F_normal)
              args.normalForceUsed= F_normal
              rospy.sleep(0.1)
              syncPub.publish(SYNC_STOP)
              rospy.sleep(0.2)

          # check if suction engage is successful
          targetPWM_Pub.publish(DUTYCYCLE_100)
          rospy.sleep(1)
          P_init = P_help.four_pressure
          args.pressure_avg = P_init
          P_vac = P_help.P_vac

          # check if suction engage is successful
          if all(np.array(P_init)[0:args.ch]<P_vac):
            print("Suction Engage Succeed!!")
            args.SuctionFlag = True
            syncPub.publish(SYNC_STOP)
          else:
            args.SuctionFlag = False

          rospy.sleep(0.1)
          targetPWM_Pub.publish(DUTYCYCLE_0)
          rospy.sleep(0.1)
                
          syncPub.publish(SYNC_STOP)
          print("Go to disengage point")
          rtde_help.goToPose(disEngagePose)
          rospy.sleep(0.5)

          # stop data logging
          rospy.sleep(0.2)
          dataLoggerEnable(False)
          rospy.sleep(0.2)
          targetPWM_Pub.publish(DUTYCYCLE_0)


          # save data and clear the temporary folder
          file_help.saveDataParams(args, appendTxt='jp_various_suction_cup_tilt_normal_'+'ch_' + str(args.ch)+'_tilt_' + str(args.tilt)+'_normal_' + str(args.normalForce)+'_material_' + str(args.material)+'_yaw_' + str(args.yaw))
          file_help.clearTmpFolder()

          if args.SuctionFlag:
            break


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
    

    # P_help.stopSampling()


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
  parser.add_argument('--normalForce', type=float, help='normal force', default= 5)
  parser.add_argument('--zHeight', type=bool, help='use presset height mode? (rather than normal force)', default= False)
  parser.add_argument('--ch', type=int, help='number of channel', default= 4)
  parser.add_argument('--material', type=int, help='MOldmax: 0, Elastic50: 1, agilus30: 2', default= 1)
  parser.add_argument('--tilt', type=int, help='tilted angle of the suction cup', default= 0)
  parser.add_argument('--yaw', type=int, help='yaw angle of the suction cup', default= 0)


  args = parser.parse_args()    
  
  main(args)
  # main(depth, rotAngleList[mode], translateZList[mode])