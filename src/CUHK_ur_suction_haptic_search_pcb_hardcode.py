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


def main(args):

  deg2rad = np.pi / 180.0
  DUTYCYCLE_100 = 100
  DUTYCYCLE_30 = 30
  DUTYCYCLE_0 = 0
  disE_x = -0.594
  disE_y = 0.100
  disE_z = 0.04
  controller_str = "W1"

  AxisAngleThres = np.pi/4.0
  disToPrevThres = 20e-3
  timeLimit = 5.0 # time limit to haptic search      
  dispLimit = 50e-3 # displacement limit = 25e-3
  angleLimit = np.pi / 4 # angle limit 45deg
  args.timeLimit = timeLimit
  args.dispLimit = dispLimit
  args.angleLimit = angleLimit

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

  # flags that reset for each attempt
  farFlag = True
  suctionSuccessFlag = False

  # try block so that we can have a keyboard exception
  try:
    
    # Go to disengage Pose
    targetPWM_Pub.publish(DUTYCYCLE_100)
    input("Press <Enter> to go disEngagePose")
    
    rtde_help.goToPose(disEngagePose)
    rospy.sleep(0.1)

    P_help.startSampling()      
    rospy.sleep(0.5)
    FT_help.setNowAsBias()
    P_help.setNowAsOffset()
    Fz_offset = FT_help.averageFz
    dataLoggerEnable(True) # start data logging


    input("Press <Enter> to go normal to get engage point")
    initEndEffPoseStamped = rtde_help.getCurrentPose()
    
    
    print("move along normal")
      #========================== Tae Change upto Here      
    targetPose = rtde_help.getCurrentPose()

    # flags and variables
    
    farFlag = True
    
    # slow approach until normal force is high enough
    F_normal = FT_help.averageFz_noOffset

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
    T_Engaged_N = np.linalg.inv(T_N_Engaged)
    engage_z = targetPoseEngaged.pose.position.z


    P_init = P_help.four_pressure
    P_vac = P_help.P_vac

    #======================== check success of initial contact ========================
    if all(np.array(P_init)<P_vac):
      print("Suction Engage Succeed from initial touch")
      suctionSuccessFlag = True # alignment flag
      print("suctionSuccessFlag: ", suctionSuccessFlag)

    # save whether initial engage was successful or not
    args.initEngageSuccess = suctionSuccessFlag

    #======================== start haptic search motion ======================
    input("Press <Enter> to start haptic search")
    PFlag = False
    startTime = time.time()
    alternateTime = time.time()
    pickStartTime = time.time()
    prevTime = 0
    
    if not suctionSuccessFlag: # and "NON" not in controller_str: # if init contact is not successful
      adpt_help.BM_step = 0 # initializa step for the Brownian motion
      while not suctionSuccessFlag:   # while no success in grasp, run controller until success or timeout
        
        # no controller
        if "NON" in controller_str:
          print("ONLY DEXNET, NO ADAPTIVE CONTROL")
          timeLimit = 2
        
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
          sequentialFailures+=1
          break

        # get FT and quat for FT control
        (trans, quat) = rtde_help.readCurrPositionQuat()
        T_array_cup = adpt_help.get_T_array_cup(T_array, F_array, quat)

        # calculate transformation matrices
        T_align, T_later = adpt_help.get_Tmats_from_controller(P_array, T_array_cup, controller_str, PFlag)
        T_normalMove = adpt_help.get_Tmat_axialMove(F_normal, F_normalThres)
        T_move =  T_later @ T_align @ T_normalMove # lateral --> align --> normal

        # move to new pose adaptively
        measuredCurrPose = rtde_help.getCurrentPose()
        currPose = adpt_help.get_PoseStamped_from_T_initPose(T_move, measuredCurrPose)
        rtde_help.goToPoseAdaptive(currPose)

        # calculate axis angle
        T_N_curr = adpt_help.get_Tmat_from_Pose(measuredCurrPose)          
        T_Engaged_curr = T_Engaged_N @ T_N_curr
        currAxisAngleToZ = np.arccos(T_N_curr[2,2])

        # calculate current pos/angle
        displacement = np.linalg.norm(T_Engaged_curr[0:3,3])
        angleDiff = np.arccos(T_Engaged_curr[2,2])

        #=================== check attempt break conditions =================== 

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
        
        # LOOP BREAK CONDITION 2
        # if timeout, or displacement/angle passed, failed
        elif time.time()-startTime >timeLimit or displacement > dispLimit or angleDiff > angleLimit or currAxisAngleToZ < (np.pi*2/3):
          args.timeOverFlag = time.time()-startTime >timeLimit
          args.dispOverFlag = displacement > dispLimit
          args.angleOverFlag = angleDiff > angleLimit
          args.angleTooFlatFlag = currAxisAngleToZ < (np.pi*2/3)

          suctionSuccessFlag = False
          print("Suction controller failed!")
          # stop at the last pose
          rtde_help.stopAtCurrPoseAdaptive()

          # keep X sec of data after alignment is complete
          rospy.sleep(0.1)
          break
        
        prevTime = time.time()



    # Save Init data
    dataLoggerEnable(False) # start data logging
    args.controller_str = controller_str
    rospy.sleep(0.5)
    file_help.saveDataParams(args,appendTxt = "JP_Lateral_searching_PCB")
    file_help.clearTmpFolder()
    P_help.stopSampling()      
    print("time to end of attempt: ", time.time() - pickStartTime)
    targetPWM_Pub.publish(DUTYCYCLE_0)


      

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

  args = parser.parse_args()    
  
  main(args)
  # main(depth, rotAngleList[mode], translateZList[mode])