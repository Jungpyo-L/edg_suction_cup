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

from keras.models import load_model
import keras.backend as kb

def mse_angular(y_true, y_pred):
    y_true = float(y_true)
    return kb.mean(kb.square(kb.minimum(kb.abs(y_pred - y_true), 360 - kb.abs(y_pred - y_true))), axis=-1)

# model_name = 'FTforGamma.h5'
# model_name = 'FTforGamma_latAmplified.h5'
# model_name = 'FTforGamma_latAmplified2.h5'
# model_name = 'FTforDomeCurvature_latAmplified2.h5'
# model_name = 'FTforPhi_latAmplified2_20000.h5'
directory = os.path.dirname(__file__)
# loaded_model =  load_model(directory + '/keras_models/' + model_name, custom_objects={'mse_angular': mse_angular})


def main(args):
  # print("directory: ", directory)
  # print("loaded_model: ", loaded_model)

  #========================== User Input================================================
  # engagePosition =  [-586e-3, 198e-3, 35e-3 - 004e-3]
  # engagePosition =  [-597e-3 - 001e-3, 200e-3, 118e-3]
  # engagePosition =  [-586e-3 + 5e-3, 198e-3, 35e-3 - 004e-3]     # for dome tilted
  # engagePosition =  [-587e-3 + 12e-3, 81e-3, 35e-3 - 004e-3]
  engagePosition =  [-587e-3 + 19e-3, 81e-3 - 10e-3, 25e-3]
  # engagePosition =  [-586e-3 + 30e-3, 198e-3, 35e-3 - 004e-3]   # for flat edge
  # engagePosition =  [-586e-3 + 29e-3, 198e-3, 35e-3 - 004e-3]   # for flat edge
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

  F_normalThres = [1.5, 1.6]
  # F_normalThres = [50, 60]
  # F_normalThres = 1.5 #1
  Fz_tolerance = 0.1
  args.domeRadius = 9999
  # args.domeRadius = 40
  # args.domeRadius = 20
  #================================================================================================
  
  # CONSTANTS
  rtde_frequency = 125

  AxisAngleThres = np.pi/4.0
  disToPrevThres = 20e-3
  timeLimit = 15.0 # time limit 10 sec      
  dispLimit = 30e-3 # displacement limit = 25e-3
  angleLimit = np.pi / 4 # angle limit 45deg
  # args.timeLimit = timeLimit
  # args.dispLimit = dispLimit
  # args.angleLimit = angleLimit

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
  adpt_help = adaptMotionHelp(dP_threshold=12, dw = 0.5, d_lat = 0.5e-3, d_z = 0.2e-3)

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

  setOrientation = tf.transformations.quaternion_from_euler(pi,0,-pi/2 -pi,'sxyz') #static (s) rotating (r)
  disEngagePose = rtde_help.getPoseObj(disengagePosition, setOrientation)
  rospy.sleep(3)
  rtde_help.goToPose(disEngagePose)

  # for i in range(0):
  #   tipContactPose = rtde_help.getCurrentPose()
  #   phi = pi/4 * 2.1
  #   omega_hat = hat(np.array([0, 0, -1]))
  #   Rw = scipy.linalg.expm(phi * omega_hat)
  #   T_from_tipContact = create_transform_matrix(Rw, [0.0, 0, 0])
  #   targetPose = adpt_help.get_PoseStamped_from_T_initPose(T_from_tipContact, tipContactPose)
  #   rtde_help.goToPose(targetPose)
  #   rospy.sleep(1)


  # pose initialization
  

  try:
    targetPWM_Pub.publish(DUTYCYCLE_0)
    
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

    # go to starting theta pose
    thetaIdx = 0
    theta = 0 * pi/180
    omega_hat1 = hat(np.array([1, 0, 0]))
    Rw1 = scipy.linalg.expm(theta * omega_hat1)

    phi = 0 * pi/180
    omega_hat2 = hat(np.array([0, 0, 1]))
    Rw2 = scipy.linalg.expm(phi * omega_hat2)

    Rw = np.dot(Rw1, Rw2)

    # L = 20e-3
    # L = 8e-3
    L = 0
    cx = L*np.sin(theta)
    # cx = 12e-3
    # cx = 0
    cz = -L*np.cos(theta)
    # cz = 0
    T_from_tipContact = create_transform_matrix(Rw, [0.0, cx, cz])

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

    # START ADAPTIVE MOTION
    input("press enter for adaptive motion")
    for i in range(5000):
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

      if F_total > 100:
        print("net force acting on cup is too high")

        # stop at the last pose
        rtde_help.stopAtCurrPose()
        rospy.sleep(0.1)
        targetPWM_Pub.publish(DUTYCYCLE_0)
        break

      # get FT and quat for FT control
      (trans, quat) = rtde_help.readCurrPositionQuat()
      T_array_cup = adpt_help.get_T_array_cup(T_array, F_array, quat)
      # FT_data = np.concatenate((T_array, F_array))
      # FT_data = np.array([T_array, F_array])
      FT_data = [[FT_help.averageFx_noOffset, FT_help.averageFy_noOffset, FT_help.averageFz_noOffset, FT_help.averageTx_noOffset, FT_help.averageTy_noOffset, FT_help.averageTz_noOffset]]
      # print("FT_data: ", FT_data)

      # calculate transformation matrices
      # controller_str = 'FTR'
      # T_align, T_later = adpt_help.get_Tmats_from_controller(P_array, T_array_cup, controller_str, PFlag)
      
      # T_align, T_later = adpt_help.get_Tmats_Suction(weightVal=1.0)
      # T_align, T_later = adpt_help.get_Tmats_freeRotation(a=0, b=1)

      # # get predicted phi from ML model
      # # phi = i
      # phi = loaded_model.predict(FT_data)
      # # print("phi: ", phi)
      # print("phi[0]: ", phi[0])

      # # +/- 90 deg to go from v to omega
      # phi_omega = (phi[0]+90) * pi/180

      # # calculate a and b from phi
      # a = -np.cos(phi[0])
      # b = np.sin(phi[0])

      # # a = -1
      # # b = 0

      # # T_align, T_later = adpt_help.get_Tmats_freeRotation(a, b)
      # T_align = np.eye(4)
      # T_later = np.eye(4)


      T_align = np.eye(4)
      T_later = np.eye(4)

      # gamma = loaded_model.predict(FT_data)
      # print("gamma: ", gamma[0])
      # if gamma < 10:    # if gamma is less than 10, then use the lateral motion only
      #   T_later = adpt_help.get_Tmat_lateralMove(P_array)
      # else:            # if gamma is greater than 10, then use rotation only
      #   T_align = adpt_help.get_Tmat_alignSuction(P_array)

      # dome = loaded_model.predict(FT_data)
      # print("dome: ", dome[0])
      # if dome < 10:    # if curavture is less than 10, then use the lateral motion only
      #   T_later = adpt_help.get_Tmat_lateralMove(P_array)
      # else:            # if curvature is greater than 10, then use rotation only
      #   T_align = adpt_help.get_Tmat_alignSuction(P_array)


      # weightVal = 1
      # T_align = adpt_help.get_Tmat_alignSuction(P_array,weightVal=weightVal )
      # T_later = adpt_help.get_Tmat_lateralMove(P_array, weightVal=1.0-weightVal)

      # weightVal = 0
      # prediction = loaded_model.predict([P_array])
      # phi = prediction[0]

      # # weightVal = 1
      # # prediction = loaded_model.predict(FT_data)
      # # ic(prediction[0])
      # # phi = -prediction[0] - 90
      # # ic(phi)
      # a = np.cos(phi)
      # b = np.sin(phi)
      # # ic(a)
      # dw = .5 * np.pi / 180.0

      # rot_axis = np.array([a,b,0])
      # norm = np.linalg.norm(rot_axis)

      # if norm == 0:
      #       # skip to checking normal force and grasp condition
      #       # continue
      #       T = np.eye(4)
      #       pass # it seems it should be pass rather than continue

      # else:     # if good, add 1 deg
      #     rot_axis = rot_axis/norm
      #     # ic(rot_axis)

      #     # publish the next target pose
      #     # print("theta: ", theta)

      #     omega_hat = hat(rot_axis)
      #     Rw = scipy.linalg.expm(dw * omega_hat)

      #     T = create_transform_matrix(Rw, [0,0,0])
      
      # T_align = T

      # weightVal = 0.25
      # T_align = adpt_help.get_Tmat_alignSuction(P_array,weightVal=weightVal )
      # T_later = adpt_help.get_Tmat_lateralMove(P_array, weightVal=1.0-weightVal)

      T_normalMove = adpt_help.get_Tmat_axialMove(F_normal, F_normalThres)
      # T_normalMove = np.eye(4)
      T_move =  T_later @ T_align @ T_normalMove # lateral --> align --> normal
      # T_move =  T_align
      # T_move = np.eye(4)

      # move to new pose adaptively
      measuredCurrPose = rtde_help.getCurrentPose()
      currPose = adpt_help.get_PoseStamped_from_T_initPose(T_move, measuredCurrPose)
      rtde_help.goToPoseAdaptive(currPose)

      reached_vacuum = all(np.array(P_array)<P_vac)

      if reached_vacuum:
        # vacuum seal formed, success!
        suctionSuccessFlag = True
        # targetPWM_Pub.publish(DUTYCYCLE_0)

        print("Suction engage succeeded with controller")

        # stop at the last pose
        rtde_help.stopAtCurrPoseAdaptive()

        # keep X sec of data after alignment is complete
        rospy.sleep(0.1)
        break


    # # initialize variables for fz control to find tipContactPose
    # print("move along normal")
    # targetPose = rtde_help.getCurrentPose()
    # rospy.sleep(.5)
    # farFlag = True
    # inRangeCounter = 0
    # F_normal = FT_help.averageFz_noOffset

    # # test data logger here
    # dataLoggerEnable(True)

    # while inRangeCounter < 100:
    #   if F_normal > -(F_normalThres-Fz_tolerance):
    #       # print("should be pushing towards surface in cup z-dir")
    #       T_normalMove = adpt_help.get_Tmat_TranlateInZ(direction = 1)
    #   elif F_normal < -(F_normalThres+Fz_tolerance):
    #       # print("should be pulling away from surface in cup z-dir")
    #       T_normalMove = adpt_help.get_Tmat_TranlateInZ(direction=-1)
    #   else:
    #       T_normalMove = np.eye(4)
    #   T_move = T_normalMove

    #   targetPose = rtde_help.getCurrentPose()
    #   currPose = adpt_help.get_PoseStamped_from_T_initPose(T_move, targetPose)
    #   rtde_help.goToPoseAdaptive(currPose)
    #   # rtde_help.goToPose(currPose)

    #   # Stop criteria
    #   F_normal = FT_help.averageFz_noOffset
    #   dF = F_normal - (-F_normalThres)
    #   # print("P_curr: ", P_curr)
    #   print("dF: ", dF)
    #   if np.abs(dF) < Fz_tolerance:
    #     inRangeCounter+= 1
    #   else:
    #     inRangeCounter = 0
    #   print(inRangeCounter)
    #   # rospy.sleep(0.05)
    
    # rtde_help.stopAtCurrPoseAdaptive()
    # rospy.sleep(0.5)
    
    # # END TEST OF DATA LOGGER HERE
    # dataLoggerEnable(False)
    # rospy.sleep(0.2)
    # P_help.stopSampling()



    # # initialize the point to sweep theta about
    # tipContactPose = rtde_help.getCurrentPose()
    # tipContactPose.pose.position.z -= 3e-3

    # # phi = 0
    # phiMax = np.pi * 2
    # phiList = np.linspace(0,phiMax,36*2+1)
    # phiList = np.array(range(0, 361, 5)) /180*np.pi
    # # phiList = np.array([0, pi/6])

    # steps = 50
    # steps = 30
    # thetaMax = steps*np.pi/180
    # thetaList = np.linspace(0,thetaMax,int(steps+1))
    # # thetaList = np.linspace( 0,thetaMax,int(steps/3+1) )
    # # thetaList = np.concatenate((np.flip(thetaList), -thetaList))
    # thetaList = np.flip(thetaList)


    # # go to starting theta pose
    # thetaIdx = 0
    # theta = 0 * pi/180
    # omega_hat1 = hat(np.array([1, 0, 0]))
    # Rw1 = scipy.linalg.expm(theta * omega_hat1)

    # phi = 0 * pi/180
    # omega_hat2 = hat(np.array([0, 0, 1]))
    # Rw2 = scipy.linalg.expm(phi * omega_hat2)

    # Rw = np.dot(Rw1, Rw2)

    # # L = 20e-3
    # L = 8e-3
    # # L = 0
    # cx = L*np.sin(theta)
    # # cx = 0
    # cz = -L*np.cos(theta)
    # # cz = 0
    # T_from_tipContact = create_transform_matrix(Rw, [0.0, cx, cz])

    # input("go to first big angle")
    # targetPose = adpt_help.get_PoseStamped_from_T_initPose(T_from_tipContact, tipContactPose)
    # rospy.sleep(0.5)
    # rtde_help.goToPose(targetPose)
    # rospy.sleep(0.5)

    

    # # START SWEEPING THETA
    # input("start sweeping theta")
    # targetPWM_Pub.publish(DUTYCYCLE_30)
    # thisThetaNeverVisited = True
    # P_vac = P_help.P_vac
    # P_curr = np.mean(P_help.four_pressure)

    # while thetaIdx < len(phiList):
    # # while thetaIdx < len(thetaList) and P_curr > P_vac:
    #   if thisThetaNeverVisited:

        
    #     P_help.startSampling()
    #     rospy.sleep(0.5)
    #     dataLoggerEnable(True) # start data logging
        

    #     # CONDITIONS
    #     phi = phiList[thetaIdx]
    #     # phi = 45
    #     theta = 20 * pi/180
    #     omega_hat1 = hat(np.array([1, 0, 0]))
    #     Rw1 = scipy.linalg.expm(theta * omega_hat1)

    #     omega_hat2 = hat(np.array([0, 0, 1]))
    #     Rw2 = scipy.linalg.expm(phi * omega_hat2)

    #     Rw = np.dot(Rw1, Rw2)
    #     # Rw = Rw1

    #     cx = L*np.sin(theta)
    #     cz = -L*np.cos(theta)

    #     # FIRST GO TO A HORIZONTAL POSITION
    #     T_from_tipContact = create_transform_matrix(Rw2, [0.0, cx, cz - 010e-3])
    #     targetPose = adpt_help.get_PoseStamped_from_T_initPose(T_from_tipContact, tipContactPose)
    #     rtde_help.goToPose(targetPose)

    #     rospy.sleep(.1)
    #     targetPWM_Pub.publish(DUTYCYCLE_100)
    #     rospy.sleep(1.5)
    #     targetPWM_Pub.publish(DUTYCYCLE_30)

    #     T_from_tipContact = create_transform_matrix(Rw, [0.0, cx, cz])
    #     targetPose = adpt_help.get_PoseStamped_from_T_initPose(T_from_tipContact, tipContactPose)
    #     rtde_help.goToPose(targetPose)
    #     thisThetaNeverVisited = False
    #     inRangeCounter = 0

    #     # P_help.startSampling()
    #     # rospy.sleep(0.5)
    #     # dataLoggerEnable(True) # start data logging
      

    #   # targetOrientation = tf.transformations.quaternion_from_euler(pi,10*pi/180,pi/2+pi/36*i,'sxyz') #static (s) rotating (r)
    #   # targetOrientation = tf.transformations.quaternion_from_euler(pi,theta /180*pi,pi/2+pi/36*i,'sxyz') #static (s) rotating (r)
    #   # targetOrientation = tf.transformations.quaternion_from_euler(pi + 30 *pi/180, 0, (90+20) *pi/180,'rxyz') #static (s) rotating (r)
    #   # targetPose = rtde_help.getPoseObj(disengagePosition, targetOrientation)
    #   # rtde_help.goToPose(targetPose)
    #   # rospy.sleep(0.1)

      

    #   P_curr = np.mean(P_help.four_pressure)
    #   F_normal = FT_help.averageFz_noOffset
    #   dF = F_normal - (-F_normalThres)

    #   if np.abs(dF) < Fz_tolerance or P_curr < P_vac:
    #     inRangeCounter+= 1
    #   else:
    #     inRangeCounter = 0
    #   print("fz counter:", inRangeCounter)
      
    #   if F_normal > -(F_normalThres-Fz_tolerance):
    #       # print("should be pushing towards surface in cup z-dir")
    #       T_normalMove = adpt_help.get_Tmat_TranlateInZ(direction = 1)
    #   elif F_normal < -(F_normalThres+Fz_tolerance):
    #       # print("should be pulling away from surface in cup z-dir")
    #       T_normalMove = adpt_help.get_Tmat_TranlateInZ(direction = -1)
    #   else:
    #       T_normalMove = np.eye(4)
    #   T_move = T_normalMove

    #   currentPose = rtde_help.getCurrentPose()
    #   targetPose_adjusted = adpt_help.get_PoseStamped_from_T_initPose(T_move, currentPose)
    #   rtde_help.goToPoseAdaptive(targetPose_adjusted)

    #   closeEnough = rtde_help.checkGoalPoseReached(targetPose_adjusted, checkDistThres=5e-3, checkQuatThres=5e-3 )
    #   # closeEnough = True

    #   # if True:
    #   # if closeEnough and np.abs(dF)<Fz_tolerance:
    #   if closeEnough and inRangeCounter > 100:
    #   # if np.abs(dF)<Fz_tolerance:

    #     rtde_help.stopAtCurrPoseAdaptive()
    #     print("Theta:", theta, "reached with Fz", F_normal)
    #     targetPWM_Pub.publish(DUTYCYCLE_100) # Just to mark in the data collection.
    #     thetaIdx+=1
    #     thisThetaNeverVisited = True
    #     rospy.sleep(2)
    #     print("2 seconds passed")
    #     targetPWM_Pub.publish(DUTYCYCLE_30) # Just to mark in the data collection.

    #     rospy.sleep(0.2)
    #     dataLoggerEnable(False) # Stop data logging
    #     rospy.sleep(0.2)  

    #     args.Fz_set = F_normalThres
    #     args.theta = int(round(theta *180/pi))
    #     args.phi = int(round(phi *180/pi))
    #     file_help.saveDataParams(args, appendTxt='seb_rotational_'+'domeRadius_' + str(args.domeRadius) + 'mm_gamma_' + str(args.theta) + '_phi_' + str(args.phi))
    #     file_help.clearTmpFolder()
    #     P_help.stopSampling()
    #     rospy.sleep(0.1)
      
    #   print("target theta: ", theta)
    #   print("P_curr: ", P_curr)
    #   print("dF: ", dF)
    #   print("np.abs(dF)<Fz_tolerance: ", np.abs(dF)<Fz_tolerance)
    #   print("checkGoalPoseReached: ", closeEnough)

    ###################################

    # save args
    args.Fz_set = F_normalThres
    args.gamma = theta
    args.phi = phi
    # args.thetaList = np.array(thetaList)
    # file_help.saveDataParams(args, appendTxt='rotCharac')
    file_help.saveDataParams(args, appendTxt='seb_testMLrotational_deleteMe')
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
    # targetPWM_Pub.publish(DUTYCYCLE_0)
    rospy.sleep(0.5)

    rospy.sleep(0.1)
    input("press enter to finish script")
    rtde_help.goToPose(disEngagePose)

    rospy.sleep(2.0)
    targetPWM_Pub.publish(DUTYCYCLE_0)

    # setOrientation = tf.transformations.quaternion_from_euler(pi,0,-pi/2 -pi,'sxyz') #static (s) rotating (r)
    # disEngagePose = rtde_help.getPoseObj(disengagePosition, setOrientation)
    # rospy.sleep(3)
    # rtde_help.goToPose(disEngagePose)

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

    # rospy.sleep(3)
    

    

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