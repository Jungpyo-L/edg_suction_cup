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
from matplotlib.animation import FuncAnimation

from moveGroupInterface_Tae import MoveGroupInterface
from scipy.io import savemat
from scipy.spatial.transform import Rotation as sciRot

# from utils import rotation_from_quaternion, create_transform_matrix, quaternion_from_matrix, normalize, hat
from helperFunction.utils import rotation_from_quaternion, create_transform_matrix, quaternion_from_matrix, normalize, hat
# from .utils import rotation_from_quaternion, create_transform_matrix, quaternion_from_matrix, normalize, hat

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
from suction_cup.srv import *
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

directory = os.path.dirname(__file__)


def main(args):
  #========================== Edge Following knobs to tune =============================

  d_lat = 1.5e-3
  d_z = 0.02e-3
  dP_threshold = 10
  P_lim = 80
  correction_scale = 0.15
  r = 35e-3

  #========================== User Input================================================
  # engagePosition =  [-586e-3, 198e-3, 35e-3 - 004e-3]
  # engagePosition =  [-597e-3 - 001e-3, 200e-3, 118e-3]
  engagePosition =  [-605e-3, 93e-3, 15e-3]     # for dome tilted
  # engagePosition =  [-586e-3 + 30e-3, 198e-3, 35e-3 - 004e-3]   # for flat edge
  # engagePosition =  [-586e-3 + 29e-3, 198e-3, 35e-3 - 004e-3]   # for flat edge
  disengagePosition = engagePosition
  disengagePosition[2] += 10e-3
  # disengagePosition[2] += 100e-3

  # controller_str = "W1"

  F_normalThres = [1.5, 1.6]
  # F_normalThres = [50, 60]
  # F_normalThres = 1.5 #1
  Fz_tolerance = 0.1
  args.domeRadius = 99999
  #================================================================================================
    # CONSTANTS
  rtde_frequency = 125

  deg2rad = np.pi / 180.0
  DUTYCYCLE_100 = 100
  DUTYCYCLE_30 = 30
  DUTYCYCLE_0 = 0

  np.set_printoptions(precision=4)
  #================================================================================================

  
  
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
  adpt_help = adaptMotionHelp(dP_threshold=0, dw = 0.5, d_lat = d_lat, d_z = d_z)
  # adpt_help = adaptMotionHelp(dP_threshold=12, dw = 0.5, d_lat = 0.5e-3, d_z = 0.2e-3)

  # SET TCP OFFSET AND CALIBRATION BETWEEN TF AND RTDE
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

  # GO TO DISENGAGE POSE
  setOrientation = tf.transformations.quaternion_from_euler(pi,0,-pi/2 -pi,'sxyz') #static (s) rotating (r)
  disEngagePose = rtde_help.getPoseObj(disengagePosition, setOrientation)

  # ROTATE IF NECESSARY
  # for i in range(0):
  #   tipContactPose = rtde_help.getCurrentPose()
  #   phi = pi/4 * 2.1
  #   omega_hat = hat(np.array([0, 0, -1]))
  #   Rw = scipy.linalg.expm(phi * omega_hat)
  #   T_from_tipContact = create_transform_matrix(Rw, [0.0, 0, 0])
  #   targetPose = adpt_help.get_PoseStamped_from_T_initPose(T_from_tipContact, tipContactPose)
  #   rtde_help.goToPose(targetPose)
  #   rospy.sleep(0.5)
  

  try:
    # go to disengage pose, save pose
    input("press enter to go to disengage pose")
    rospy.sleep(0.1)
    rtde_help.goToPose(disEngagePose)
    targetPWM_Pub.publish(DUTYCYCLE_30)

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
    rospy.sleep(0.5)

    tipContactPose = rtde_help.getCurrentPose()

    # SET GAMMA AND PHI FOR ROTATED CONFIG
    theta = 0 * pi/180
    omega_hat1 = hat(np.array([1, 0, 0]))
    Rw1 = scipy.linalg.expm(theta * omega_hat1)

    phi = 0 * pi/180
    omega_hat2 = hat(np.array([0, 0, 1]))
    Rw2 = scipy.linalg.expm(phi * omega_hat2)

    Rw = np.dot(Rw1, Rw2)

    # CALCULATE HOW BEHIND WE MOVE BACKWARDS
    # L = 20e-3
    # L = 8e-3
    L = 0
    # cx = L*np.sin(theta)
    # cx = 6e-3
    # cx = 13e-3
    # cx = -10e-3
    cx = 0
    cz = -L*np.cos(theta)
    # cz = 0
    T_from_tipContact = create_transform_matrix(Rw, [0.0, cx, cz])

    # GO TO BACK POSE
    input("go to first big angle")
    targetPose = adpt_help.get_PoseStamped_from_T_initPose(T_from_tipContact, tipContactPose)
    rospy.sleep(0.5)
    rtde_help.goToPose(targetPose)
    rospy.sleep(0.5)
    targetPWM_Pub.publish(DUTYCYCLE_100)

    # set up flags for adaptive motion
    PFlag = False
    startTime = time.time()
    alternateTime = time.time()
    prevTime = 0
    P_vac = P_help.P_vac

    targetPoseStamped = copy.deepcopy(targetPose)

    # START CIRCULAR TRAJECTORY
    targetPWM_Pub.publish(DUTYCYCLE_30)
    input("press enter to press down")

    farFlag = True
    F_normal = FT_help.averageFz_noOffset

    # input('approach for Fz control')
    while farFlag:
      ic(F_normal)
      rospy.sleep(0.010)
      if F_normal > -F_normalThres[0]:
        T_move = adpt_help.get_Tmat_TranlateInZ(direction = 1)
        targetPoseStamped = adpt_help.get_PoseStamped_from_T_initPose(T_move, targetPoseStamped)
        rtde_help.goToPoseAdaptive(targetPoseStamped, time = 0.1)

        # new normal force
        F_normal = FT_help.averageFz_noOffset

      elif F_normal < -F_normalThres[1]:
        T_move = adpt_help.get_Tmat_TranlateInZ(direction = -1)
        targetPoseStamped = adpt_help.get_PoseStamped_from_T_initPose(T_move, targetPoseStamped)
        rtde_help.goToPoseAdaptive(targetPoseStamped, time = 0.1)
        
        # new normal force
        F_normal = FT_help.averageFz_noOffset

      else:
        farFlag = False
        rtde_help.stopAtCurrPoseAdaptive()
        # print("reached threshhold normal force: ", F_normal)
        rospy.sleep(0.01)
    
    measuredCurrPose = rtde_help.getCurrentPose()
    rospy.sleep(0.01)

    # once we reach the normal force, we can start the circular trajectory
    targetPWM_Pub.publish(DUTYCYCLE_30)
    input("press enter to start circular trajectory")
    
    # find the next point to move to, and go to it
    edgeFollowing_history = []

    circle_resolution = int(3600/2)
    thetas = np.linspace(1,360,circle_resolution)

    x = r*np.sin(thetas*deg2rad)
    y = r*np.cos(thetas*deg2rad)

    ic(x)
    ic(y)

    num_iterations = circle_resolution
    startPose = copy.deepcopy(measuredCurrPose)
    # nextPose = copy.deepcopy(measuredCurrPose)

    i = 0
    if i == 0:

      xi = x[i]
      yi = y[i]
      ic([xi, yi])

      T_from_startPose = create_transform_matrix(Rw, [xi, yi, 0])
      currentPose = adpt_help.get_PoseStamped_from_T_initPose(T_from_startPose, startPose)
      rtde_help.goToPoseAdaptive(currentPose)  

      rospy.sleep(2.01)
      targetPWM_Pub.publish(DUTYCYCLE_100)

      farFlag = True
      F_normal = FT_help.averageFz_noOffset

      # input('approach for Fz control')
      while farFlag:
        ic(F_normal)
        rospy.sleep(0.010)
        if F_normal > -F_normalThres[0]:
          T_move = adpt_help.get_Tmat_TranlateInZ(direction = 1)
          currentPose = adpt_help.get_PoseStamped_from_T_initPose(T_move, currentPose)
          rtde_help.goToPoseAdaptive(currentPose, time = 0.1)

          # new normal force
          F_normal = FT_help.averageFz_noOffset

        elif F_normal < -F_normalThres[1]:
          T_move = adpt_help.get_Tmat_TranlateInZ(direction = -1)
          currentPose = adpt_help.get_PoseStamped_from_T_initPose(T_move, currentPose)
          rtde_help.goToPoseAdaptive(currentPose, time = 0.1)
          
          # new normal force
          F_normal = FT_help.averageFz_noOffset

        else:
          farFlag = False
          rtde_help.stopAtCurrPoseAdaptive()
          # print("reached threshhold normal force: ", F_normal)
          rospy.sleep(0.01)
          startPose = rtde_help.getCurrentPose()
          currentPose = rtde_help.getCurrentPose()

    history_edgeFollowing = []
    startTime = time.time()
    for i in range(1,num_iterations):

      ic(F_normal)
      farFlag = True
      F_normal = FT_help.averageFz_noOffset

      # input('approach for Fz control')
      while farFlag:
        ic(F_normal)
        rospy.sleep(0.010)
        if F_normal > -F_normalThres[0]:
          T_move = adpt_help.get_Tmat_TranlateInZ(direction = 1)
          currentPose = adpt_help.get_PoseStamped_from_T_initPose(T_move, currentPose)
          rtde_help.goToPoseAdaptive(currentPose, time = 0.1)

          # new normal force
          F_normal = FT_help.averageFz_noOffset

        elif F_normal < -F_normalThres[1]:
          T_move = adpt_help.get_Tmat_TranlateInZ(direction = -1)
          currentPose = adpt_help.get_PoseStamped_from_T_initPose(T_move, currentPose)
          rtde_help.goToPoseAdaptive(currentPose, time = 0.1)
          
          # new normal force
          F_normal = FT_help.averageFz_noOffset

        else:
          farFlag = False
          rtde_help.stopAtCurrPoseAdaptive()
          # print("reached threshhold normal force: ", F_normal)
          rospy.sleep(0.01)
          currentPose = rtde_help.getCurrentPose()

      # ic(currentPose)
      height = currentPose.pose.position.z

      # T_from_tipContact = create_transform_matrix(Rw, [0.0, 0.0001, cz])
      # nextPose = adpt_help.get_PoseStamped_from_T_initPose(T_from_tipContact, nextPose)
      xi = x[i]
      yi = y[i] - r
      thti = thetas[i]
      timei = time.time() - startTime
      P_array = P_help.four_pressure
      # ic([xi, yi])

      T_from_startPose = create_transform_matrix(Rw, [xi, yi, 0])
      currentPose = adpt_help.get_PoseStamped_from_T_initPose(T_from_startPose, startPose)
      currentPose.pose.position.z = height
      rtde_help.goToPoseAdaptive(currentPose)
      currentPose = rtde_help.getCurrentPose()
      
      rospy.sleep(0.01)

      history_edgeFollowing.append([timei, thti, xi, yi, F_normal, *P_array])
      ic(history_edgeFollowing[i-1])


    ###################################
    ic(history_edgeFollowing)
    
    # save args
    args.edgeFollowing_history = history_edgeFollowing
    args.Fz_set = F_normalThres
    args.phi = phi
    args.gamma = theta
    args.r = r
    # args.thetaList = np.array(thetaList)
    # file_help.saveDataParams(args, appendTxt='rotCharac')
    file_help.saveDataParams(args, appendTxt='seb_testEdgeFollowing_hardcodedCircle_deleteMe')
    # file_help.saveDataParams(args, appendTxt='seb_rotational_'+'domeRadius_' + str(args.domeRadius) + '_gamma_' + str(args.gamma) + '_theta_' + str(args.theta))
    # file_help.saveDataParams(args, appendTxt='jp_lateral_'+'xoffset_' + str(args.xoffset)+'_theta_' + str(args.theta))
    file_help.clearTmpFolder()
    P_help.stopSampling()
    rospy.sleep(0.5)

    # stop logger, stop sampling, and save data
    rtde_help.stopAtCurrPoseAdaptive()
    rospy.sleep(0.5)
    dataLoggerEnable(False) # start data logging
    rospy.sleep(0.5)      
    P_help.stopSampling()
    targetPWM_Pub.publish(DUTYCYCLE_0)
    rospy.sleep(0.5)

    rospy.sleep(0.1)
    input("press enter to finish script")
    rtde_help.goToPose(disEngagePose)

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
    rospy.sleep(.5)

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