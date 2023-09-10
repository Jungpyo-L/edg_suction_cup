#!/usr/bin/env python

# Authors: Sebastian D. Lee, Jungpyo Lee, and Tae Myung Huh

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

import endeffectorOffset as eff_offsetCal
from tae_psoc.msg import cmdToPsoc
from tae_psoc.msg import SensorPacket

from netft_utils.srv import *
from tae_datalogger.srv import *
from std_msgs.msg import String
from std_msgs.msg import Int8
from geometry_msgs.msg import WrenchStamped
from geometry_msgs.msg import PoseStamped
import geometry_msgs.msg
import tf
import cv2
from scipy import signal

from math import pi, cos, sin
from controller_manager_msgs.msg import ControllerState
from controller_manager_msgs.srv import *
from controller_manager_msgs.utils\
    import ControllerLister, ControllerManagerLister,\
    get_rosparam_controller_names


from dynamic_reconfigure.srv import *
from dynamic_reconfigure.msg import Config

from helperFunction.SuctionP_callback_helper import P_CallbackHelp
from helperFunction.FT_callback_helper import FT_CallbackHelp
from helperFunction.fileSaveHelper import fileSaveHelp
from helperFunction.rtde_helper import rtdeHelp
from helperFunction.adaptiveMotion import adaptMotionHelp
from helperFunction.gqcnn_policy_class import GraspProcessor


def main(args):

  deg2rad = np.pi / 180.0
  DUTYCYCLE_100 = 100
  DUTYCYCLE_30 = 30
  DUTYCYCLE_0 = 0
  disE_x = -0.593
  disE_y = 0.220
  disE_z = 0.025

  F_normalThres = [1.5, 2.0]

  FT_SimulatorOn = False
  np.set_printoptions(precision=4)

  # controller node
  rospy.init_node('jp_ur_run')

  # Setup helper functions
  FT_help = FT_CallbackHelp() # it deals with subscription.
  rospy.sleep(0.5)
  P_help = P_CallbackHelp() # it deals with subscription.
  rospy.sleep(0.5)
  rtde_help = rtde_help = rtdeHelp(125)
  rospy.sleep(0.5)
  file_help = fileSaveHelp()
  adpt_help = adaptMotionHelp(dw = 0.5, d_lat = 0.5e-3, d_z = 0.1e-3)

  # Load Camera Transform matrix
  
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


  print("Wait for the data_logger to be enabled")
  rospy.wait_for_service('data_logging')
  dataLoggerEnable = rospy.ServiceProxy('data_logging', Enable)
  dataLoggerEnable(False) # reset Data Logger just in case
  rospy.sleep(1)
  file_help.clearTmpFolder()        # clear the temporary folder

  
  # pose initialization
  xoffset = args.xoffset
  disengagePosition =  [disE_x + args.xoffset, disE_y, disE_z] # When depth is 0 cm. unit is in m
  setOrientation = tf.transformations.quaternion_from_euler(pi,0,pi/2,'sxyz') #static (s) rotating (r)
  disEngagePose = rtde_help.getPoseObj(disengagePosition, setOrientation)


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
    initEndEffPoseStamped = rtde_help.getCurrentPose()
    
    
    print("move along normal")
      #========================== Tae Change upto Here      
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
          rospy.sleep(0.1)
    
    targetPoseEngaged = rtde_help.getCurrentPose()
    T_N_Engaged = adpt_help.get_Tmat_from_Pose(targetPoseEngaged)
    engage_z = targetPoseEngaged.pose.position.z
    rtde_help.goToPose(disEngagePose)
    rospy.sleep(1)    
    
    input("Press <Enter> to start data collection")
    
    for j in range (0,1): #range(4): #range(16,17): #range(25):
      args.xoffset = -j*0.001 # to go oposite direction
      disengagePosition =  [disE_x + args.xoffset, disE_y, disE_z]
      engagePosition = [disE_x + args.xoffset, disE_y, engage_z]
      # for i in range(round(args.angle/5)+1):
      for i in range(0, 1):
        args.angle = 0
        print("offset: ", args.xoffset)
        print("Pose Idx: ", i)
        args.theta = round((pi/36*i)*180/pi)
        print("Theta =", args.theta)

        if args.startAngle > args.theta:
          continue
        
        for k in range(0, 11):
          # args.speed = 0.3 + k*0.05
          args.acc = round(0.2 + k*0.1, 1)

          targetPWM_Pub.publish(DUTYCYCLE_100)
          # rotate
          targetOrientation = tf.transformations.quaternion_from_euler(pi,0,pi/2+pi/36*i,'sxyz') #static (s) rotating (r)
          targetPose = rtde_help.getPoseObj(disengagePosition, targetOrientation)
          rtde_help.goToPose(targetPose, speed = args.speed, acc = args.acc)
          rospy.sleep(0.1)
          

          P_help.startSampling()
          rospy.sleep(0.5)
          dataLoggerEnable(True) # start data logging
          rospy.sleep(0.5)
          
          # move down
          targetOrientation = tf.transformations.quaternion_from_euler(pi,0,pi/2+pi/36*i,'sxyz') #static (s) rotating (r)
          targetPose = rtde_help.getPoseObj(engagePosition, targetOrientation)
          rtde_help.goToPose(targetPose, speed = args.speed, acc = args.acc)
          rospy.sleep(0.1)
          
          # input("Press <Enter> to move along normal")
          
          # collect the init 
          
          rospy.sleep(1.5)
          P_init = P_help.four_pressure
          P_vac = P_help.P_vac
          if all(np.array(P_init)<P_vac) and i == 0:
            print("Suction Engage Succeed from initial touch")
            targetPWM_Pub.publish(DUTYCYCLE_0)
            SuctionFlag = True
          else:
            SuctionFlag = False

          # move up
          targetOrientation = tf.transformations.quaternion_from_euler(pi,0,pi/2+pi/36*i,'sxyz') #static (s) rotating (r)
          targetPose = rtde_help.getPoseObj(disengagePosition, targetOrientation)
          rtde_help.goToPose(targetPose, speed = args.speed, acc = args.acc)
          rospy.sleep(0.1)

          targetPWM_Pub.publish(DUTYCYCLE_0)
          dataLoggerEnable(False) # stop data logging
          rospy.sleep(0.2)
          

          file_help.saveDataParams(args, appendTxt='jp_lateral_'+'xoffset_' + str(args.xoffset)+'_theta_' + str(args.theta)+'_speed_' + str(args.speed)+'_acc_' + str(args.acc))
          file_help.clearTmpFolder()
          P_help.stopSampling()
          rospy.sleep(0.5)
          if SuctionFlag == True:
            break
      
      if SuctionFlag == False:
        for i in range(round(args.angle/90)): 
          targetOrientation = tf.transformations.quaternion_from_euler(pi,0,pi/2 + args.angle*pi/180-pi/2*(i+1),'sxyz') #static (s) rotating (r)
          targetPose = rtde_help.getPoseObj(disengagePosition, targetOrientation)
          rtde_help.goToPose(targetPose)
          rospy.sleep(0.1)      
    

    print("Go to disengage point")
    disengagePosition =  [disE_x, disE_y, disE_z] # When depth is 0 cm. unit is in m
    setOrientation = tf.transformations.quaternion_from_euler(pi,0,pi/2,'sxyz') #static (s) rotating (r)
    disEngagePose = rtde_help.getPoseObj(disengagePosition, setOrientation)
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
  parser.add_argument('--acc', type=float, help='Acceleration of UR10 robot', default= 0.2)
  parser.add_argument('--speed', type=float, help='Linear speed of UR10 robot', default= 0.3)

  args = parser.parse_args()
  
  main(args)
  # main(depth, rotAngleList[mode], translateZList[mode])