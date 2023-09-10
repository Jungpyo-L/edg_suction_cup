#!/usr/bin/env python

# Authors: Sebastian D. Lee and Tae Myung Huh and Jungpyo Lee

# imports
try:
  import rospy
  import tf
  ros_enabled = True
except:
  print('Couldn\'t import ROS.  I assume you\'re running this on your laptop')
  ros_enabled = False


import rospy
import tf
ros_enabled = True

from calendar import month_abbr
import os, sys
import string
import matplotlib.pyplot as plt


from moveGroupInterface_Tae import MoveGroupInterface
from scipy.io import savemat
from scipy.spatial.transform import Rotation as sciRot

# from utils import rotation_from_quaternion, create_transform_matrix, quaternion_from_matrix, normalize, hat
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
  #========================== User Input================================================
  engagePosition =  [-576e-3, 198e-3, 35e-3]
  disengagePosition = engagePosition
  disengagePosition[2] += 10e-3
  # disengagePosition[2] += 100e-3

  # controller_str = "NON"
  # controller_str = "W1"
  controller_str = "W2"
  # controller_str = "W5"
  # controller_str = "FTR"
  # controller_str = "PRLalt"
  # controller_str = "FTRPL"

  # F_normalThres = [1.5, 2.0]
  F_normalThres = 1.5 #1
  Fz_tolerance = 0.1
  #================================================================================================
  
  # CONSTANTS
  rtde_frequency = 125

  AxisAngleThres = np.pi/4.0
  disToPrevThres = 20e-3
  timeLimit = 15.0 # time limit 10 sec      
  dispLimit = 30e-3 # displacement limit = 25e-3
  angleLimit = np.pi / 4 # angle limit 45deg
  args.timeLimit = timeLimit
  args.dispLimit = dispLimit
  args.angleLimit = angleLimit

  deg2rad = np.pi / 180.0
  DUTYCYCLE_100 = 100
  DUTYCYCLE_30 = 30
  DUTYCYCLE_0 = 0

  np.set_printoptions(precision=4)
  
  # CONTROLLER NODE
  rospy.init_node('tae_ur_run')

  # INITIALIZE HELPER FUNCTION OBJECTS
  FT_help = FT_CallbackHelp() # it deals with subscription.
  rospy.sleep(0.5)
  P_help = P_CallbackHelp() # it deals with subscription.
  rospy.sleep(0.5)
  rtde_help = rtdeHelp(rtde_frequency)
  rospy.sleep(0.5)
  file_help = fileSaveHelp()
  adpt_help = adaptMotionHelp(dw = 0.5, d_lat = 0.5e-3, d_z = 0.2e-3)

  # set tcp offset and calibration between tf and rtde
  rospy.sleep(0.5)
  rtde_help.setTCPoffset([0, 0, 0.146, 0, 0, 0])
  rospy.sleep(0.2)
  rtde_help.setCalibrationMatrix()
  rospy.sleep(0.2)

  # Set the PWM Publisher, pwm0
  targetPWM_Pub = rospy.Publisher('pwm', Int8, queue_size=1)
  targetPWM_Pub.publish(DUTYCYCLE_0)

  # enable data logger
  print("Wait for the data_logger to be enabled")
  rospy.wait_for_service('data_logging')
  dataLoggerEnable = rospy.ServiceProxy('data_logging', Enable)
  dataLoggerEnable(False) # reset Data Logger just in case
  rospy.sleep(1.0)
  file_help.clearTmpFolder()        # clear the temporary folder

  # pose initialization
  setOrientation = tf.transformations.quaternion_from_euler(pi,0,pi/2,'sxyz') #static (s) rotating (r)
  disEngagePose = rtde_help.getPoseObj(disengagePosition, setOrientation)

  try:

    # go to disengage pose, save pose
    targetPWM_Pub.publish(DUTYCYCLE_0)
    rospy.sleep(0.1)
    rtde_help.goToPose(disEngagePose)
    input("press enter")

    # start sampling pressure
    P_help.startSampling()
    rospy.sleep(0.5)
    
    # set biases now until it works
    biasNotSet = True
    while biasNotSet:
      try:
        FT_help.setNowAsBias()
        rospy.sleep(0.1)
        P_help.setNowAsOffset()
        rospy.sleep(0.1)
        biasNotSet = False
      except:
        print("set now as offset failed, but it's okay")

    # start data logger
    dataLoggerEnable(True) # start data logging

    # initialize variables for fz control to find tipContactPose
    print("move along normal")
    targetPose = rtde_help.getCurrentPose()
    rospy.sleep(.5)
    farFlag = True
    inRangeCounter = 0
    F_normal = FT_help.averageFz_noOffset

    while inRangeCounter < 100:
      if F_normal > -(F_normalThres-Fz_tolerance):
          # print("should be pushing towards surface in cup z-dir")
          T_normalMove = adpt_help.get_Tmat_TranlateInZ(direction = 1)
      elif F_normal < -(F_normalThres+Fz_tolerance):
          # print("should be pulling away from surface in cup z-dir")
          T_normalMove = adpt_help.get_Tmat_TranlateInZ(direction=-1)
      else:
          T_normalMove = np.eye(4)
      T_move = T_normalMove

      targetPose = rtde_help.getCurrentPose()
      currPose = adpt_help.get_PoseStamped_from_T_initPose(T_move, targetPose)
      rtde_help.goToPoseAdaptive(currPose)
      # rtde_help.goToPose(currPose)

      # Stop criteria
      F_normal = FT_help.averageFz_noOffset
      dF = F_normal - (-F_normalThres)
      # print("P_curr: ", P_curr)
      print("dF: ", dF)
      if np.abs(dF) < Fz_tolerance:
        inRangeCounter+= 1
      else:
        inRangeCounter = 0
      print(inRangeCounter)
      # rospy.sleep(0.05)
    
    rtde_help.stopAtCurrPoseAdaptive()
    rospy.sleep(0.5)

    # initialize the point to sweep theta about
    tipContactPose = rtde_help.getCurrentPose()
    tipContactPose.pose.position.z -= 3e-3

    phi = 0
    phiMax = np.pi / 2
    phiList = np.linspace(0,phiMax,11)

    # steps = 50
    steps = 30
    thetaMax = steps*np.pi/180
    thetaList = np.linspace(0,thetaMax,int(steps+1))
    # thetaList = np.linspace( 0,thetaMax,int(steps/3+1) )
    # thetaList = np.concatenate((np.flip(thetaList), -thetaList))
    thetaList = np.flip(thetaList)

    # go to starting theta pose
    thetaIdx=0
    theta = thetaList[thetaIdx]
    omega_hat = hat(np.array([np.cos(phi), np.sin(phi), 0]))
    Rw = scipy.linalg.expm(theta * omega_hat)

    # L = 20e-3
    L = 8e-3
    # L = 0
    cx = L*np.sin(theta)
    # cx = 0
    cz = -L*np.cos(theta)
    # cz = 0
    T_from_tipContact = create_transform_matrix(Rw, [0.0, cx, cz])

    input("go to first big angle")
    targetPose = adpt_help.get_PoseStamped_from_T_initPose(T_from_tipContact, tipContactPose)
    rospy.sleep(0.5)
    rtde_help.goToPose(targetPose)
    rospy.sleep(0.5)

    input("start sweeping theta")
    targetPWM_Pub.publish(DUTYCYCLE_100)

    # sweep the rotation offsets
    thisThetaNeverVisited = True
    P_vac = P_help.P_vac
    P_curr = np.mean(P_help.four_pressure)

    while thetaIdx < len(thetaList):
    # while thetaIdx < len(thetaList) and P_curr > P_vac:
      if thisThetaNeverVisited:
        theta = thetaList[thetaIdx]
        omega_hat = hat(np.array([np.cos(phi), np.sin(phi), 0]))
        Rw = scipy.linalg.expm(theta * omega_hat)
        # T_from_tipContact = create_transform_matrix(Rw, [0.0, 0.0, 0.0])
        cx = L*np.sin(theta)
        cz = -L*np.cos(theta)
        T_from_tipContact = create_transform_matrix(Rw, [0.0, cx, cz])
        targetPose = adpt_help.get_PoseStamped_from_T_initPose(T_from_tipContact, tipContactPose)
        rtde_help.goToPose(targetPose)
        thisThetaNeverVisited = False
        inRangeCounter = 0
      

      i = 2
      targetOrientation = tf.transformations.quaternion_from_euler(pi,0,pi/2+pi/36*i,'sxyz') #static (s) rotating (r)
      targetOrientation = tf.transformations.quaternion_from_euler(pi,theta /180*pi,pi/2+pi/36*i,'sxyz') #static (s) rotating (r)
      targetPose = rtde_help.getPoseObj(disengagePosition, targetOrientation)
      rtde_help.goToPose(targetPose)
      rospy.sleep(0.1)

      P_curr = np.mean(P_help.four_pressure)
      F_normal = FT_help.averageFz_noOffset
      dF = F_normal - (-F_normalThres)

      if np.abs(dF) < Fz_tolerance or P_curr < P_vac:
        inRangeCounter+= 1
      else:
        inRangeCounter = 0
      print("fz counter:", inRangeCounter)
      
      if F_normal > -(F_normalThres-Fz_tolerance):
          # print("should be pushing towards surface in cup z-dir")
          T_normalMove = adpt_help.get_Tmat_TranlateInZ(direction = 1)
      elif F_normal < -(F_normalThres+Fz_tolerance):
          # print("should be pulling away from surface in cup z-dir")
          T_normalMove = adpt_help.get_Tmat_TranlateInZ(direction = -1)
      else:
          T_normalMove = np.eye(4)
      T_move = T_normalMove

      currentPose = rtde_help.getCurrentPose()
      targetPose_adjusted = adpt_help.get_PoseStamped_from_T_initPose(T_move, currentPose)
      rtde_help.goToPoseAdaptive(targetPose_adjusted)

      closeEnough = rtde_help.checkGoalPoseReached(targetPose_adjusted, checkDistThres=5e-3, checkQuatThres=5e-3 )
      # closeEnough = True

      # if True:
      # if closeEnough and np.abs(dF)<Fz_tolerance:
      if closeEnough and inRangeCounter > 100:
      # if np.abs(dF)<Fz_tolerance:
        rtde_help.stopAtCurrPoseAdaptive()
        print("Theta:", theta, "reached with Fz", F_normal)
        targetPWM_Pub.publish(DUTYCYCLE_100) # Just to mark in the data collection.
        thetaIdx+=1
        thisThetaNeverVisited = True
        rospy.sleep(4)
        print("4 seconds passed")
        # rospy.sleep(0.5)
      
      print("target theta: ", theta)
      print("P_curr: ", P_curr)
      print("dF: ", dF)
      print("np.abs(dF)<Fz_tolerance: ", np.abs(dF)<Fz_tolerance)
      print("checkGoalPoseReached: ", closeEnough)

    ###########################
    # # start data logger
    # dataLoggerEnable(True) # start data logging

    # set up flags for adaptive motion
    PFlag = False
    startTime = time.time()
    alternateTime = time.time()
    prevTime = 0

    for i in range(0):
      print(i)
      # for alternating controller
      if time.time() - alternateTime > 0.5:
        alternateTime = time.time()
        if PFlag:
          PFlag = False
        else:
          PFlag = True
      
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
        targetPWM_Pub.publish(DUTYCYCLE_0)
        break

      # get FT and quat for FT control
      (trans, quat) = rtde_help.readCurrPositionQuat()
      T_array_cup = adpt_help.get_T_array_cup(T_array, F_array, quat)

      # calculate transformation matrices
      # controller_str = 'FTR'
      # T_align, T_later = adpt_help.get_Tmats_from_controller(P_array, T_array_cup, controller_str, PFlag)
      
      T_align, T_later = adpt_help.get_Tmats_Suction(weightVal=1.0)
      # T_align, T_later = adpt_help.get_Tmats_freeRotation(a=0, b=1)

      # T_normalMove = adpt_help.get_Tmat_axialMove(F_normal, F_normalThres)
      T_normalMove = np.eye(4)
      T_move =  T_later @ T_align @ T_normalMove # lateral --> align --> normal
      # T_move =  T_align
      # T_move = np.eye(4)

      # move to new pose adaptively
      measuredCurrPose = rtde_help.getCurrentPose()
      currPose = adpt_help.get_PoseStamped_from_T_initPose(T_move, measuredCurrPose)
      rtde_help.goToPoseAdaptive(currPose)

    ###################################

    # stop logger, stop sampling, and save data
    rtde_help.stopAtCurrPoseAdaptive()
    rospy.sleep(0.5)
    dataLoggerEnable(False) # start data logging
    rospy.sleep(0.5)      
    P_help.stopSampling()
    targetPWM_Pub.publish(DUTYCYCLE_0)
    rospy.sleep(0.5)
    # rtde_help.goToPoseAdaptive(disEngagePose, time = 1.0)
    rtde_help.goToPose(disEngagePose)

    # save args
    args.Fz_set = F_normalThres
    args.thetaList = np.array(thetaList)
    # file_help.saveDataParams(args, appendTxt='rotCharac')
    file_help.saveDataParams(args, appendTxt='seb_rotational_'+'domeRadius_' + str(args.domeRadius) + '_gamma_' + str(args.gamma) + '_theta_' + str(args.theta))
    # file_help.saveDataParams(args, appendTxt='jp_lateral_'+'xoffset_' + str(args.xoffset)+'_theta_' + str(args.theta))
    file_help.clearTmpFolder()

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
  parser.add_argument('--attempIndex', type=int, help='startIndex Of pick attempt', default= 1)

  args = parser.parse_args()    
  
  main(args)
  # main(depth, rotAngleList[mode], translateZList[mode])