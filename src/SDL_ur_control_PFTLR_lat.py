#!/usr/bin/env python

# Authors: Sebastian D. Lee and Tae Myung Huh and Jungpyo Lee

# test change

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
from icecream import ic


def main(args):
  #========================== User Input================================================
  # flat-edge-tilt
  engagePosition =  [-(500e-3 - 059e-3), 200e-3 + 049e-3, 24e-3]
  args.domeRadius = 9999
  args.edge = 1

  offset = -13
  args.offset = offset
  engagePosition[0] += offset/1000
  
  # # flat-tilt
  # engagePosition[0] -= 040e-3
  # engagePosition[1] -= 040e-3
  # args.domeRadius = 9999
  # args.edge = 0
  
  # dome-tilt
  engagePosition[1] += 040e-3
  args.domeRadius = 20
  args.edge = 0

  disengagePosition = engagePosition.copy()
  disengagePosition[2] += 1e-3

  F_normalThres = 1.5 #1
  F_lim = 1.5
  Fz_tolerance = 0.1
  #================================================================================================
  
  # CONSTANTS
  rtde_frequency = 125

  DUTYCYCLE_100 = 100
  DUTYCYCLE_30 = 30
  DUTYCYCLE_0 = 0

  np.set_printoptions(precision=3)
  
  # CONTROLLER NODE
  rospy.init_node('tae_ur_run')

  # INITIALIZE HELPER FUNCTION OBJECTS
  FT_help = FT_CallbackHelp() # it deals with subscription.
  rospy.sleep(0.1)
  P_help = P_CallbackHelp() # it deals with subscription.
  rospy.sleep(0.1)
  rtde_help = rtdeHelp(rtde_frequency)
  rospy.sleep(0.1)
  file_help = fileSaveHelp()
  adpt_help = adaptMotionHelp(dw = 0.5, d_lat = 0.5e-3, d_z = 0.2e-3)

  # set tcp offset and calibration between tf and rtde
  rospy.sleep(0.1)
  # rtde_help.setTCPoffset([0, 0, 0.146, 0, 0, 0])    # original for TRO
  rtde_help.setTCPoffset([0, 0, 0.156, 0, 0, 0])      # new for testing
  rospy.sleep(0.1)

  # Set the PWM Publisher, pwm0
  targetPWM_Pub = rospy.Publisher('pwm', Int8, queue_size=1)
  rospy.sleep(0.1)
  targetPWM_Pub.publish(DUTYCYCLE_0)

  # enable data logger
  print("Wait for the data_logger to be enabled")
  rospy.wait_for_service('data_logging')
  dataLoggerEnable = rospy.ServiceProxy('data_logging', Enable)
  dataLoggerEnable(False) # reset Data Logger just in case
  rospy.sleep(1.0)
  file_help.clearTmpFolder()        # clear the temporary folder

  setOrientation = tf.transformations.quaternion_from_euler(pi,0,-pi/2 -pi,'sxyz') #static (s) rotating (r)
  engagePose = rtde_help.getPoseObj(engagePosition, setOrientation)
  disEngagePose = rtde_help.getPoseObj(disengagePosition, setOrientation)

  # pose initialization
  rospy.sleep(0.1)
  # rtde_help.goToPose(disEngagePose)
  rtde_help.goToPose(engagePose)

  # # for reseting yaw angle
  # for i in range(0):
  #   tipContactPose = rtde_help.getCurrentPose()
  #   phi = pi/4 * 2.1
  #   omega_hat = hat(np.array([0, 0, -1]))
  #   Rw = scipy.linalg.expm(phi * omega_hat)
  #   T_from_tipContact = create_transform_matrix(Rw, [0.0, 0, 0])
  #   targetPose = adpt_help.get_PoseStamped_from_T_initPose(T_from_tipContact, tipContactPose)
  #   rtde_help.goToPose(targetPose)
  #   rospy.sleep(1)


  try:

    input("press enter to go to engage pose")

    # go to disengage pose, then egage pose
    targetPWM_Pub.publish(DUTYCYCLE_30)
    rospy.sleep(0.1)
    rtde_help.goToPose(disEngagePose)
    rospy.sleep(0.1)
    rtde_help.goToPose(engagePose)
    rospy.sleep(0.1)

    # start sampling pressure
    P_help.startSampling()
    rospy.sleep(0.1)
    
    # set biases now until it works
    biasNotSet = True
    while biasNotSet:
      try:
        FT_help.setNowAsBias()
        rospy.sleep(0.05)
        P_help.setNowAsOffset()
        rospy.sleep(0.05)
        biasNotSet = False
      except:
        print("set now as offset failed, but it's okay")

    # # START TEST OF DATA LOGGER HERE
    # # dataLoggerEnable(True) # start data logging

    # # initialize variables for fz control to find tipContactPose
    # print("move along normal")
    # # targetPose = rtde_help.getCurrentPose()
    # rospy.sleep(.5)
    # farFlag = True
    # inRangeCounter = 0
    # F_normal = FT_help.averageFz_noOffset

    # # test data logger here
    # dataLoggerEnable(True)

    # # while inRangeCounter < 100:
    # #   if F_normal > -(F_normalThres-Fz_tolerance):
    # #       # print("should be pushing towards surface in cup z-dir")
    # #       T_normalMove = adpt_help.get_Tmat_TranlateInZ(direction = 1)
    # #   elif F_normal < -(F_normalThres+Fz_tolerance):
    # #       # print("should be pulling away from surface in cup z-dir")
    # #       T_normalMove = adpt_help.get_Tmat_TranlateInZ(direction=-1)
    # #   else:
    # #       T_normalMove = np.eye(4)
    # #   T_move = T_normalMove

    # #   targetPose = rtde_help.getCurrentPose()
    # #   currPose = adpt_help.get_PoseStamped_from_T_initPose(T_move, targetPose)
    # #   rtde_help.goToPoseAdaptive(currPose)
    # #   # rtde_help.goToPose(currPose)

    # #   # Stop criteria
    # #   F_normal = FT_help.averageFz_noOffset
    # #   dF = F_normal - (-F_normalThres)
    # #   # print("P_curr: ", P_curr)
    # #   print("dF: ", dF)
    # #   if np.abs(dF) < Fz_tolerance:
    # #     inRangeCounter+= 1
    # #   else:
    # #     inRangeCounter = 0
    # #   print(inRangeCounter)
    # #   # rospy.sleep(0.05)
    
    # # rtde_help.stopAtCurrPoseAdaptive()
    # # rospy.sleep(0.5)
    
    # # END TEST OF DATA LOGGER HERE
    # dataLoggerEnable(False)
    # rospy.sleep(0.2)
    # P_help.stopSampling()

    # initialize the point to sweep theta about
    tipContactPose = rtde_help.getCurrentPose()
    tipContactPose.pose.position.z -= 0e-3
    rtde_help.goToPose(tipContactPose)

    # SET PHI
    phi = 0
    # phiMax = np.pi * 2
    # phiList = np.linspace(0,phiMax,36*2+1)
    # phiList = np.array(range(0, 361, 5)) / 180*np.pi
    # # phiList = np.array([0, pi/6])
    # # phiList = np.array([0])
    # # phiList = np.array(range(0, 46, 5)) / 180*np.pi

    
    # SET THETA
    # steps = 50
    # steps = 30
    # thetaMax = steps*np.pi/180
    # thetaList = np.linspace(0,thetaMax,int(steps+1))
    # # thetaList = np.linspace( 0,thetaMax,int(steps/3+1) )
    # # thetaList = np.concatenate((np.flip(thetaList), -thetaList))
    # thetaList = np.flip(thetaList)
    # thetaList = -np.array(range(0, 46, 5)) / 180*np.pi
    thetaList = np.array([0,0])

    # go to starting theta pose
    thetaIdx = 0
    theta = 0 * pi/180
    omega_hat1 = hat(np.array([1, 0, 0]))
    Rw1 = scipy.linalg.expm(theta * omega_hat1)

    phi = 0 * pi/180
    omega_hat2 = hat(np.array([0, 0, 1]))
    Rw2 = scipy.linalg.expm(phi * omega_hat2)

    Rw = np.dot(Rw1, Rw2)

    r_cup = 11.5e-3 + 2e-3

    # L = 3e-3
    # L = 8e-3
    # L = 12e-3
    L = r_cup*np.sin(theta) + 2e-3
    # L = 0
    cx = L*np.sin(theta)
    # cx = 0
    cz = -L*np.cos(theta)
    # cz = 0
    T_from_tipContact = create_transform_matrix(Rw, [0.0, cx, cz])

    input("go to first big angle")
    targetPose = adpt_help.get_PoseStamped_from_T_initPose(T_from_tipContact, tipContactPose)
    rospy.sleep(0.1)
    rtde_help.goToPose(targetPose)
    rospy.sleep(0.8)

    input("start sweeping theta")
    targetPWM_Pub.publish(DUTYCYCLE_30)

    # sweep the rotation offsets
    thisThetaNeverVisited = True
    P_vac = P_help.P_vac
    P_curr = np.mean(P_help.four_pressure)

    # while thetaIdx < len(thetaList):
    while thetaIdx < len(thetaList)-1:
    # while thetaIdx < len(np.array([0,1])):
    # while thetaIdx < len(thetaList) and P_curr > P_vac:
      if thisThetaNeverVisited:

        # START sampling and logging data
        P_help.startSampling()
        rospy.sleep(0.5)
        dataLoggerEnable(True) # start data logging
        

        # CONDITIONS
        phi = 0
        # phi = 45
        # theta = 45 * pi/180
        theta = 0 
        print("theta: ", theta/pi*180)
        theta = thetaList[thetaIdx]
        omega_hat1 = hat(np.array([1, 0, 0]))
        Rw1 = scipy.linalg.expm(theta * omega_hat1)

        omega_hat2 = hat(np.array([0, 0, 1]))
        Rw2 = scipy.linalg.expm(phi * omega_hat2)

        Rw = np.dot(Rw1, Rw2)
        # Rw = Rw1

        # L = r_cup*np.sin(np.abs(theta)) + 2e-3
        # cx = L*np.sin(theta)
        # cz = -L*np.cos(np.abs(theta))
        cx = 0e-3
        cz = -2e-3

        # FIRST GO TO A HORIZONTAL POSITION with chosen phi value (Rw2)
        T_from_tipContact = create_transform_matrix(Rw2, [0.0, cx, cz])
        targetPose = adpt_help.get_PoseStamped_from_T_initPose(T_from_tipContact, tipContactPose)
        rtde_help.goToPose(targetPose)

        rospy.sleep(0.5)
        targetPWM_Pub.publish(DUTYCYCLE_100)
        rospy.sleep(1.5)
        targetPWM_Pub.publish(DUTYCYCLE_0)

        # THEN GO TO THE DESIRED ROTATED STATE (Rw)
        T_from_tipContact = create_transform_matrix(Rw, [0.0, cx, cz])
        targetPose = adpt_help.get_PoseStamped_from_T_initPose(T_from_tipContact, tipContactPose)
        rtde_help.goToPose(targetPose)

        rospy.sleep(0.5)
        targetPWM_Pub.publish(DUTYCYCLE_100)
        rospy.sleep(1.5)
        targetPWM_Pub.publish(DUTYCYCLE_0)

        thisThetaNeverVisited = False
        inRangeCounter = 0

        rospy.sleep(0.5)
        targetPWM_Pub.publish(DUTYCYCLE_100)

      
      # ONCE IN THE ROTATED STATE, MOVE ALONG Z-AXIS

      # Get current PFT
      P_curr = np.mean(P_help.four_pressure)
      F_normal = FT_help.averageFz_noOffset

      # count criteria
      dF = F_normal - (-F_normalThres)
      if np.abs(dF) < Fz_tolerance or P_curr < P_vac:
        inRangeCounter+= 1
      else:
        inRangeCounter = 0
      print("fz counter:", inRangeCounter)

      # determine direction to move along z-axis
      if F_normal > -(F_normalThres-Fz_tolerance):
          # print("should be pushing towards surface in cup z-dir")
          T_normalMove = adpt_help.get_Tmat_TranlateInZ(direction = 1)
      elif F_normal < -(F_normalThres+Fz_tolerance):
          # print("should be pulling away from surface in cup z-dir")
          T_normalMove = adpt_help.get_Tmat_TranlateInZ(direction = -1)
      else:
          T_normalMove = np.eye(4)
      T_move = T_normalMove

      # move from current pose adaptively
      currentPose = rtde_help.getCurrentPose()
      targetPose_adjusted = adpt_help.get_PoseStamped_from_T_initPose(T_move, currentPose)
      rtde_help.goToPoseAdaptive(targetPose_adjusted)

      # closeEnough = rtde_help.checkGoalPoseReached(targetPose_adjusted, checkDistThres=5e-3, checkQuatThres=5e-3 )
      # closeEnough = True

      # if True:
      # if closeEnough and np.abs(dF)<Fz_tolerance:
      # if closeEnough and inRangeCounter > 100:
      if inRangeCounter > 100:  
      # if np.abs(dF)<Fz_tolerance:

        rtde_help.stopAtCurrPoseAdaptive()
        rospy.sleep(0.1)
        targetPWM_Pub.publish(DUTYCYCLE_0)
        rospy.sleep(0.5)
        print("Theta:", theta, "reached with Fz", F_normal)
        targetPWM_Pub.publish(DUTYCYCLE_100) # Just to mark in the data collection.
        thetaIdx+=1
        thisThetaNeverVisited = True
        rospy.sleep(1.5)
        print("1 second passed")
        targetPWM_Pub.publish(DUTYCYCLE_0) # Just to mark in the data collection.

        rospy.sleep(0.2)
        dataLoggerEnable(False) # Stop data logging
        rospy.sleep(0.2)  

        args.Fz_set = F_normalThres
        args.gamma = int(round(theta *180/pi))
        args.phi = int(round(phi *180/pi))
        file_help.saveDataParams(args, appendTxt='seb_rotational_'+'domeRadius' + str(args.domeRadius) + 'mm_gamma' + str(args.gamma) + '_phi' + str(args.phi) + '_edge' + str(args.edge) + '_offset' + str(args.offset))
        file_help.clearTmpFolder()
        P_help.stopSampling()
        rospy.sleep(0.1)
      
      print("target theta: ", theta/pi*180)
      print("P_curr: ", P_curr)
      print("dF: ", dF)
      print("np.abs(dF)<Fz_tolerance: ", np.abs(dF)<Fz_tolerance)
      # print("checkGoalPoseReached: ", closeEnough)

    ###################################

    # stop logger, stop sampling, and save data
    rtde_help.stopAtCurrPoseAdaptive()
    rospy.sleep(0.1)
    dataLoggerEnable(False) # start data logging
    rospy.sleep(0.1)      
    P_help.stopSampling()
    targetPWM_Pub.publish(DUTYCYCLE_0)
    rospy.sleep(0.1)

    rospy.sleep(0.1)
    input("press enter to finish script")

    setOrientation = tf.transformations.quaternion_from_euler(pi,0,-pi/2 -pi,'sxyz') #static (s) rotating (r)
    disEngagePose = rtde_help.getPoseObj(disengagePosition, setOrientation)
    rospy.sleep(0.1)
    rtde_help.goToPose(disEngagePose)
    rospy.sleep(0.1)

    # for i in range(2):
    #   tipContactPose = rtde_help.getCurrentPose()
    #   phi = pi/4 * 2.1
    #   omega_hat = hat(np.array([0, 0, -1]))
    #   Rw = scipy.linalg.expm(phi * omega_hat)
    #   T_from_tipContact = create_transform_matrix(Rw, [0.0, 0, 0])
    #   targetPose = adpt_help.get_PoseStamped_from_T_initPose(T_from_tipContact, tipContactPose)
    #   rtde_help.goToPose(targetPose)
    #   rospy.sleep(1)
    # rtde_help.goToPose(disEngagePose)

    rospy.sleep(0.1)
    

    # save args
    # args.Fz_set = F_normalThres
    # args.thetaList = np.array(thetaList)
    # # file_help.saveDataParams(args, appendTxt='rotCharac')
    # file_help.saveDataParams(args, appendTxt='seb_rotational_'+'domeRadius_' + str(args.domeRadius) + '_gamma_' + str(args.gamma) + '_theta_' + str(args.theta))
    # # file_help.saveDataParams(args, appendTxt='jp_lateral_'+'xoffset_' + str(args.xoffset)+'_theta_' + str(args.theta))
    # file_help.clearTmpFolder()

    print("============ Python UR_Interface demo complete!")

    

  except rospy.ROSInterruptException:
    print('rospy.ROSInterruptException')
    return
  except KeyboardInterrupt:
    print(KeyboardInterrupt)

    phi = -pi
    omega_hat = hat(np.array([0, 0, 1]))
    Rw = scipy.linalg.expm(phi * omega_hat)
    T_from_tipContact = create_transform_matrix(Rw, [0.0, cx, cz - 020e-3])
    targetPose = adpt_help.get_PoseStamped_from_T_initPose(T_from_tipContact, tipContactPose)
    rtde_help.goToPose(targetPose)
    rospy.sleep(0.1)

    setOrientation = tf.transformations.quaternion_from_euler(pi,0,pi/2,'sxyz') #static (s) rotating (r)
    disEngagePose = rtde_help.getPoseObj(disengagePosition, setOrientation)

    rtde_help.goToPose(disEngagePose)

    return


if __name__ == '__main__':  
  import argparse
  parser = argparse.ArgumentParser()
  # parser.add_argument('--useStoredData', type=bool, help='take image or use existingFile', default=False)
  # parser.add_argument('--storedDataDirectory', type=str, help='location of target saved File', default="")
  # parser.add_argument('--attempIndex', type=int, help='startIndex Of pick attempt', default= 1)

  args = parser.parse_args()    
  
  main(args)
  # main(depth, rotAngleList[mode], translateZList[mode])