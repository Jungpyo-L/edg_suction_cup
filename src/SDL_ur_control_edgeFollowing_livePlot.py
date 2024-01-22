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

import keras
import keras.backend as kb
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, BatchNormalization, Lambda

def mse_angular(y_true, y_pred):
    y_true = float(y_true)
    return kb.mean(kb.square(kb.minimum(kb.abs(y_pred - y_true), 360 - kb.abs(y_pred - y_true))), axis=-1)

directory = os.path.dirname(__file__)
output_scale = 360.0
model_name = 'PforPhi_Data_amplified_thresh.h5'
loaded_model = load_model(directory+ '/keras_models/' + model_name, custom_objects={'mse_angular': mse_angular})

def main(args):
  #========================== Edge Following knobs to tune =============================

  d_lat = 4.0e-3
  d_z = 0.02e-3
  dP_threshold = 10
  P_lim_upper = 400
  P_lim_lower = 200
  correction_scale = 0.15

  #========================== User Input================================================
  # engagePosition =  [-586e-3, 198e-3, 35e-3 - 004e-3]
  # engagePosition =  [-597e-3 - 001e-3, 200e-3, 118e-3]
  # engagePosition =  [-574e-3, 90e-3, 15e-3]     # for dome tilted
  # engagePosition =  [-574e-3 -66e-3, 90e-3, 15e-3]     # for dome tilted
  engagePosition =  [-605e-3, 98e-3, 15e-3]   # hard coded circle
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
    # cx = 0
    cx = 33e-3
    # cz = -L*np.cos(theta)
    cz = 0
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

    # START ADAPTIVE MOTION
    input("press enter for adaptive motion")

    num_iterations = 10  # You can replace this with the actual number of iterations in your loop
    time_values = np.zeros(num_iterations)
    r_p_values = np.zeros((num_iterations, 2))

    farFlag = True
    F_normal = FT_help.averageFz_noOffset

    # input('approach for Fz control')
    while farFlag:
      ic(F_normal)
      rospy.sleep(0.01)
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

    edgeFollowing_history = []
    # history_edgeFollowing = []
    for i in range(80000):
      # print(i)

      farFlag = True
      F_normal = FT_help.averageFz_noOffset

      # # input('approach for Fz control')
      # while farFlag:
      #   ic(F_normal)
      #   rospy.sleep(0.01)
      #   if F_normal > -F_normalThres[0]:
      #     T_move = adpt_help.get_Tmat_TranlateInZ(direction = 1)
      #     targetPoseStamped = adpt_help.get_PoseStamped_from_T_initPose(T_move, targetPoseStamped)
      #     rtde_help.goToPoseAdaptive(targetPoseStamped, time = 0.1)

      #     # new normal force
      #     F_normal = FT_help.averageFz_noOffset

      #   elif F_normal < -F_normalThres[1]:
      #     T_move = adpt_help.get_Tmat_TranlateInZ(direction = -1)
      #     targetPoseStamped = adpt_help.get_PoseStamped_from_T_initPose(T_move, targetPoseStamped)
      #     rtde_help.goToPoseAdaptive(targetPoseStamped, time = 0.1)
          
      #     # new normal force
      #     F_normal = FT_help.averageFz_noOffset

      #   else:
      #     farFlag = False
      #     rtde_help.stopAtCurrPoseAdaptive()
      #     # print("reached threshhold normal force: ", F_normal)
      #     rospy.sleep(0.01)
      
      measuredCurrPose = rtde_help.getCurrentPose()
      rospy.sleep(0.01)
      
      # for alternating controller
      if time.time() - alternateTime > 0.5:
        alternateTime = time.time()
        if PFlag:
          PFlag = False
        else:
          PFlag = True
      
      # PFT arrays to calculate Transformation matrices
      currentTime = time.time() - startTime
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

      # # get FT and quat for FT control
      # (trans, quat) = rtde_help.readCurrPositionQuat()
      # T_array_cup = adpt_help.get_T_array_cup(T_array, F_array, quat)
      # # FT_data = np.concatenate((T_array, F_array))
      # # FT_data = np.array([T_array, F_array])
      # FT_data = [[FT_help.averageFx_noOffset, FT_help.averageFy_noOffset, FT_help.averageFz_noOffset, FT_help.averageTx_noOffset, FT_help.averageTy_noOffset, FT_help.averageTz_noOffset]]

      T_align = np.eye(4)
      T_later = np.eye(4)

      # pressure data
      P0, P1, P2, P3 = P_array
      PW = (P3 + P2)/2
      PE = (P1 + P0)/2
      PN = (P1 + P2)/2
      PS = (P0 + P3)/2

      # pressure differentials
      dP_WE = PW - PE        # 0 deg
      dP_SN = PS - PN        # 90 deg
      dP_NW_SE = P2 - P0     # 45 deg
      dP_SW_NE = P3 - P1     # -45 deg

      # direction vector
      r_p = np.zeros(2)  
      r_p[0] = -dP_SN
      r_p[1] = dP_WE
      r_p = r_p / np.linalg.norm(r_p)

      # correction vector
      corr_p = np.zeros(2)
      corr_p[0] = -dP_WE
      corr_p[1] = -dP_SN
      corr_p = corr_p / np.linalg.norm(corr_p)

      # flip correction vector depending on absolute pressure reading
      # ic(np.mean(P_array))
      # ic(max(dP_WE,dP_SN))
      if abs(np.mean(P_array)) > P_lim_upper:
        corr_p = -corr_p
      
      if abs(np.mean(P_array)) < P_lim_upper and abs(np.mean(P_array)) > P_lim_lower:
        corr_p = np.zeros(2)

      P_mean = abs(np.mean(P_array))
      P_mean_log10 = np.log10(P_mean)
      # ic(P_mean_log10)

      # sum of direction vector and correction vector
      correction_scale = P_mean_log10 / 4
      # correction_scale = 0.65
      # correction_scale = 0
      r_p_final = r_p + corr_p * correction_scale

      ic(correction_scale)

      if False:
        prediction = loaded_model.predict([P_array])
        phi = prediction[0]
        ic(phi)
        # r_p_final = np.array([np.cos(phi), np.sin(phi)])
      
      # normalize again
      r_p_final = r_p_final / np.linalg.norm(r_p_final)

      # compute the lateral movements
      dx_lat = 0.0
      dy_lat = 0.0
      if abs(dP_WE) > dP_threshold or abs(dP_SN) > dP_threshold:
        dy_lat = r_p_final[0] * d_lat
        dx_lat = r_p_final[1] * d_lat

      T_later = adpt_help.get_Tmat_TranlateInBodyF([dx_lat, dy_lat, 0.0])

      T_normalMove = adpt_help.get_Tmat_axialMove(F_normal, F_normalThres)
      # T_normalMove = np.eye(4)
      T_move =  T_later @ T_align @ T_normalMove # lateral --> align --> normal
      # T_move =  T_normalMove
      # T_move =  T_align
      # T_move = np.eye(4)
      # ic(np.sqrt(Fx**2 + Fy**2))
      # if np.sqrt(Fx**2 + Fy**2)<0.1:
      #   T_move =  T_normalMove

      xi = measuredCurrPose.pose.position.x
      yi = measuredCurrPose.pose.position.y

      # edgeFollowing_history.append([currentTime, r_p, corr_p, r_p_final, F_normal, *P_array])
      edgeFollowing_history.append([currentTime, xi, yi, r_p, corr_p, r_p_final, F_normal, *P_array])
      # history_edgeFollowing.append([timei, thti, xi, yi, F_normal, *P_array])

      if not 'fig' in locals():
        fig, ax = plt.subplots(figsize=(8, 4))

      # Clear the previous plot and create a new one
      ax.clear()
      # ax.scatter(r_p[0], r_p[1], alpha=0.3, color='#FF0000', lw=2, ec='red')
      # ax.scatter(corr_p[0], corr_p[1], alpha=0.3, color='#FF0000', lw=1, ec='black')
      # ax.scatter(r_p_final[0], r_p_final[1], alpha=0.3, color='#FF0000', lw=1, ec='blue')

      arrow_properties1 = dict(facecolor='black', edgecolor='black', alpha=0.7, width=0.5, headwidth=8)
      arrow_properties2 = dict(facecolor='black', edgecolor='black', alpha=0.7, width=3.0, headwidth=12)
      ax.annotate('', xytext=(0,0), xy=(-r_p_final[0], -r_p_final[1]), arrowprops=arrow_properties2)
      ax.annotate('', xytext=(0,0), xy=(-r_p[0], -r_p[1]), arrowprops=arrow_properties1)
      ax.annotate('', xytext=(0,0), xy=(-corr_p[0] * correction_scale, -corr_p[1] * correction_scale), arrowprops=arrow_properties1)

      # lims = [0, 5]
      lims = [-1.0, 1.0]
      ax.axvline(0, color='blue', linestyle='--', linewidth=1)
      ax.axhline(0, color='blue', linestyle='--', linewidth=1)

      # ax.plot(lims, lims, lw=1, color='#0000FF')
      ax.ticklabel_format(useOffset=False, style='plain')
      ax.tick_params(axis='both', which='major', labelsize=18)
      ax.set_xlim(lims)
      ax.set_ylim(lims)
      # ax.set_xlim([-1, 1])
      # ax.set_ylim([-1, 1])
      ax.set_aspect('equal', adjustable='box')
      ax.set_title(f'Loop i: {i}')

      # Add a pause to allow the figure to update
      plt.pause(0.01)


      # move to new pose adaptively
      measuredCurrPose = rtde_help.getCurrentPose()
      # currPose = adpt_help.get_PoseStamped_from_T_initPose(T_move, measuredCurrPose)
      # rtde_help.goToPoseAdaptive(currPose)
      targetPoseStamped = adpt_help.get_PoseStamped_from_T_initPose(T_move, measuredCurrPose)
      targetSearchPoseStamped = copy.deepcopy(targetPoseStamped)
      rtde_help.goToPoseAdaptive(targetSearchPoseStamped)
      rospy.sleep(0.01)

      reached_vacuum = all(np.array(P_array)<P_vac)

      if reached_vacuum:
        # vacuum seal formed, success!
        suctionSuccessFlag = True
        targetPWM_Pub.publish(DUTYCYCLE_0)

        print("Suction engage succeeded with controller")

        # stop at the last pose
        rtde_help.stopAtCurrPoseAdaptive()

        # keep X sec of data after alignment is complete
        rospy.sleep(0.1)
        break

      if abs(P_mean) < dP_threshold/2:
        # vacuum seal formed, success!
        suctionSuccessFlag = False
        targetPWM_Pub.publish(DUTYCYCLE_0)

        print("went off edge")

        # stop at the last pose
        rtde_help.stopAtCurrPoseAdaptive()

        # keep X sec of data after alignment is complete
        rospy.sleep(0.1)
        break


    ###################################
    # ic(edgeFollowing_history)
    
    # save args
    args.edgeFollowing_history = edgeFollowing_history
    args.Fz_set = F_normalThres
    args.phi = phi
    args.gamma = theta
    # args.thetaList = np.array(thetaList)
    # file_help.saveDataParams(args, appendTxt='rotCharac')
    file_help.saveDataParams(args, appendTxt='seb_testEdgeFollowing_deleteMe')
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