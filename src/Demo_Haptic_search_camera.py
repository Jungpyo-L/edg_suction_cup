#!/usr/bin/env python

# Authors: Sebastian D. Lee and Tae Myung Huh and Jungpyo Lee

# library imports
# try:
import rospy
import tf
ros_enabled = True

from calendar import month_abbr
import os, sys
import string
import matplotlib.pyplot as plt


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
from icecream import ic

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
# except:
#   print('Couldn\'t import ROS.  I assume you\'re running this on your laptop')
#   ros_enabled = False

from keras.models import load_model

# model_name = 'FTforGamma.h5'
# model_name = 'FTforGamma_latAmplified.h5'
# model_name = 'FTforGamma_latAmplified2.h5'
model_name = 'FTforDomeCurvature_latAmplified2.h5'
directory = os.path.dirname(__file__)
loaded_model =  load_model(directory + '/keras_models/' + model_name)

def main(args):
  #========================== User Input================================================
  H_cup = 420e-3

  disengagePosition =  [-690e-3, 65e-3, H_cup]
  
  TopLeftPix = [50, 50] # x, y  # Need to double check from the test image.
  BottomRightPix = [640-50, 480-50] # x, y
  # boxWidth = 240e-3
  # boxLength = 300e-3
  BoxTopLeftCorner_meter = [-480e-3, 270e-3, 70e-3] # for new bin picking configuration (230125)
  BoxBottomRightCorner_meter = [-830e-3, -220e-3, 70e-3] # for new bin picking configuration (230125)
  dropBoxPosition = [-1030e-3, 25e-3, H_cup - 50e-3]

  # controller_str = "NON"
  # controller_str = "W1"
  controller_str = "W2"
  # controller_str = "W3"
  # controller_str = "W4"
  # controller_str = "W5"
  # controller_str = "FTR"
  # controller_str = "PRLalt"
  # controller_str = "FTRPL"
  # controller_str = "BML"
  # controller_str = "BMR"
  # controller_str = "BMLR"
  # controller_str = "DomeCurv"

  F_normalThres = [2.2, 2.5]

  #================================================================================================
  
  # CONSTANTS
  h_cam_cup = 80e-3
  thickness = 10e-3
  # thickness = 100e-3
  depthThreshold = h_cam_cup + H_cup - thickness
  depthThreshold = 495e-3
  # depthThreshold = 100e-3

  fileAppendixStr = 'GQCNN_ADAPTIVE'
  pick_place_z = H_cup
  rtde_frequency = 125
  dt = 1.0/rtde_frequency

  AxisAngleThres = np.pi/4.0
  disToPrevThres = 1e-3
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

  FT_SimulatorOn = False
  GQ_CNN_SIMUL_On = False
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
  adpt_help = adaptMotionHelp(dw = 0.5, d_lat = 0.5e-3, d_z = 0.1e-3)
  policyGen = GraspProcessor(depth_thres = depthThreshold, num_actions=30)

  # Load Camera Transform matrix    
  datadir = os.path.dirname(os.path.realpath(__file__))     # datadir = os.path.expanduser('~') + "/catkin_ws_new/src/tae_ur_experiment/src/"
  # with open(datadir + '/TransformMat_board_verified', 'rb') as handle:
  with open(datadir + '/TransformMat_board_verified_bin_picking', 'rb') as handle:
      loaded_data = pickle.load(handle)
  T_cam = loaded_data
  print("T_cam: ", T_cam)

  # set tcp offset and calibration between tf and rtde
  rospy.sleep(0.5)
  rtde_help.setTCPoffset([0, 0, 0.146, 0, 0, 0])
  rospy.sleep(0.2)
  rtde_help.setCalibrationMatrix()
  rospy.sleep(0.2)

  # if FT_SimulatorOn:
  #   print("wait for FT simul")
  #   rospy.wait_for_service('start_sim')
  #   # bring the service
  #   netftSimCall = rospy.ServiceProxy('start_sim', StartSim)

  # Set the PWM Publisher  
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
  setOrientation = tf.transformations.quaternion_from_euler(pi,0,0,'sxyz') #static (s) rotating (r)
  dropBoxOrientation = tf.transformations.quaternion_from_euler(pi,pi/3,0,'sxyz') #static (s) rotating (r)
  disEngagePose = rtde_help.getPoseObj(disengagePosition, setOrientation)
  BoxTopLeftCornerPose = rtde_help.getPoseObj(BoxTopLeftCorner_meter, setOrientation)
  BoxBottomRightCornerPose = rtde_help.getPoseObj(BoxBottomRightCorner_meter, setOrientation)
  dropBoxPose = rtde_help.getPoseObj(dropBoxPosition, dropBoxOrientation)

  # initilize counters
  failedPicks = 0
  noViablePoint_count = 0

  # try block so that we can have a keyboard exception
  try:
    # initialize memory system: T_targetPose_prev and 
    T_targetPose_prev = np.zeros((3,3))
    Dist_fromPrevTest = np.zeros(3)

    if args.useStoredData:
      datadir = file_help.ResultSavingDirectory

      with open(datadir + '/targetPointList.p', 'rb') as handle:
        targetPointList = pickle.load(handle)

      with open(datadir + '/prevTrialInfo.p', 'rb') as handle:
        prevTrialInfo = pickle.load(handle)

      T_targetPose_prev = targetPointList
      attemptIdx = prevTrialInfo[0]
      successfulPicks = prevTrialInfo[1]
      sequentialFailures = prevTrialInfo[2]

    else:
      for i in range(3):
        T_targetPose_prev[i] = disengagePosition
      attemptIdx = 0
      successfulPicks = 0
      sequentialFailures = 0
    
      pickHistory = []
      
    
    while attemptIdx < 5 and successfulPicks < 19 and sequentialFailures < 10:

      input("Press <Enter> to go start Haptic Search")

      # =================== Check status of experiment, then take new photo, get poses ===================
      pickStartTime = time.time()
      
      print("Attempt: ", attemptIdx)
      print("successfulPicks: ", successfulPicks)
      print("failures in a row: ", sequentialFailures)
      
      # input("Press <Enter> to go disEngagePose")
      print("goToPose disEngagePose")
      ultimateSuccess = 0
      ic(pickHistory)
      rtde_help.goToPose(disEngagePose)
      rospy.sleep(0.5)
      initEndEffPoseStamped = rtde_help.getCurrentPose()

      # input("Press <Enter> to take photo")
      print('take photo')
      print("time since start of pick after photo: ", time.time() - pickStartTime)
      
      # Take Image from realSense
      policyGen.depth_im = None
      print('taking image with realsense and getting grasp points')
      policyGen.getGQCNN_Grasps_Bin(fileAppendix='_'+fileAppendixStr+str(attemptIdx), TopLeftPix=TopLeftPix, BottomRightPix=BottomRightPix) # take preset # of grasp point
      # policyGen.getGQCNN_Grasps()
      print("time after getGQCNN_Grasps_Bin: ", time.time() - pickStartTime)
      print("getting surface normals from pointcloud")
      # policyGen.getSurfaceNormals_PointCloud(zMin=depthThreshold-0.15, zMax = depthThreshold, visualizeOn=False, vectorNum = 3) # vec Num is high because we will skip the points that are close
      
      # input("press enter, check for CPU")

      # save poses
      GQCNN_poses = policyGen.plannedGrasps
      GQCNN_poseQVals = policyGen.plannedGraspsQvals
      poseList = GQCNN_poses
      selectedQVal = np.nan

      # input("press enter, check for CPU")

      # Copy transformation matrix
      currDir_path = os.path.dirname(os.path.realpath(__file__))    
      shutil.copyfile(os.path.join(currDir_path, 'TransformMat_board_verified_bin_picking'), os.path.join(policyGen.ResultSavingDirectory, 'TransformMat_board_verified_bin_picking'))
      args.storedDataDirectory = policyGen.ResultSavingDirectory

      # flags that reset for each attempt
      farFlag = True
      suctionSuccessFlag = False
      

      # =================== Check if the highest priority pose is viable ===================
      for poseIdx in range(0, len(poseList) ):
        
        skipPrev = False
        print("Pose Idx:= ", poseIdx)
        args.thisPoseIdx = poseIdx
        thisPose = poseList[poseIdx]
        # print("thisPose: ", thisPose)
        
        targetPoseStamped = adpt_help.getGoalPosestampedFromGQCNN(T_cam, thisPose, initEndEffPoseStamped)
        T_targetPose = adpt_help.get_Tmat_from_Pose(targetPoseStamped)
        rospy.sleep(0.1)
        try:
          intersecPoint = adpt_help.intersection_between_cup_box(T_targetPose, BoxTopLeftCorner_meter, BoxBottomRightCorner_meter)
        except:
          print("error from intersecPoint")
          continue
        print("intersecPoint: ", intersecPoint)
        # print("BoxTopLeftCorner_meter[2]: ", BoxTopLeftCorner_meter[2])

        currAxisAngleToZ = np.arccos(T_targetPose[2,2])
        if currAxisAngleToZ < (np.pi-AxisAngleThres): # if the angle is > 45deg
          print("Skipping this pose becuase of angle: ", str(currAxisAngleToZ))        
          continue

        # memory system to avoid going near the same spot after a failure
        # memorize three previous points
        for i in range(3):
          Dist_fromPrevTest[i] = np.linalg.norm(T_targetPose_prev[i] - [T_targetPose[0:3,3]])

        # if the target pose is too close to previous, skip the run.
        for distance in Dist_fromPrevTest:
          if distance < disToPrevThres:
            print("Skipping this pose becuase of distance")        
            skipPrev = True
            break
        if skipPrev:
          continue
        T_targetPose_prev[:-1] = T_targetPose_prev[1:]
        T_targetPose_prev[-1] = T_targetPose[0:3,3]
        
        if intersecPoint[2] > BoxTopLeftCorner_meter[2]: # z Threshold
          targetPoseStamped
          selectedQVal = GQCNN_poseQVals[poseIdx]
          noViablePoint_count = 0
          print("selectedIdx and Qval: ", poseIdx, selectedQVal)
          break
      
      # rechecking that q value exists 
      if np.isnan(selectedQVal):
        noViablePoint_count +=1
        if noViablePoint_count > 1:
          print("Stuck because there's no viable point, break out after 5")
          break # it is stuck, get out of the while loop
        else:
          continue # recapture the image and rerun GQCNN

      #=================== get ready to move axially to attempt grasp ===================   
      
      print("time after finding highest priority pose: ", time.time() - pickStartTime)
      T_offset = adpt_help.get_Tmat_TranlateInBodyF([0., 0., -15e-3]) # small offset from the target pose
      targetSearchPoseStamped = adpt_help.get_PoseStamped_from_T_initPose(T_offset, targetPoseStamped)

      # input("Press <Enter> to move targetSearchPoseStamped")
      print("Moving to StartEngagePoint")
      rtde_help.goToPose(targetSearchPoseStamped)
      rospy.sleep(0.5)

      # start sampling pressure and FT data, bias both
      P_help.startSampling()
      rospy.sleep(0.5)
      
      # set biases now
      try:
        FT_help.setNowAsBias()
        rospy.sleep(0.1)
        P_help.setNowAsOffset()
        rospy.sleep(0.1)
      except:
        print("set now as offset failed, but it's okay")

      # start data logger
      dataLoggerEnable(True) # start data logging

      #=================== move axially ===================  
      # input("Press <Enter> to move along normal")
      print("move along normal")
      targetPose = rtde_help.getCurrentPose()
      rospy.sleep(0.5)
      F_normal = FT_help.averageFz_noOffset

      targetPWM_Pub.publish(DUTYCYCLE_0)

      # input("press enter, check for CPU and axial movement")
 
      print("time before axial movement: ", time.time() - pickStartTime)
      # while cup is far from object
      rospy.sleep(3)
      for i in range(3):
        T_move = adpt_help.get_Tmat_TranlateInZ(direction = 1)
        targetPose = adpt_help.get_PoseStamped_from_T_initPose(T_move, targetPose)
        rtde_help.goToPoseAdaptive(targetPose, time = 0.1)
      
      rospy.sleep(1)
      

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
      
      print("time after axial movement: ", time.time() - pickStartTime)
      # start vacuum
      targetPWM_Pub.publish(DUTYCYCLE_100)

      # get current pose to check limits
      targetPoseEngaged = rtde_help.getCurrentPose()
      T_N_Engaged = adpt_help.get_Tmat_from_Pose(targetPoseEngaged) # relative angle limit
      T_Engaged_N = np.linalg.inv(T_N_Engaged)
      
      #=================== check success of gqcnn ===================  
      # compared pressure readings to vacuum pressure, sets the success flag 
      P_init = P_help.four_pressure
      P_vac = P_help.P_vac
      if all(np.array(P_init)<P_vac):
        print("Suction Engage Succeed from initial touch")
        suctionSuccessFlag = True # alignment flag
        print("suctionSuccessFlag: ", suctionSuccessFlag)
        # initialize T_targetPose_prev
        for i in range(3):
          T_targetPose_prev[i] = disengagePosition
      
      # save whether initial engage was successful or not
      args.initEngageSuccess = suctionSuccessFlag

      #=================== start adaptive motion if gqcnn failed ===================  
      # flag for alternating, time that controller starts trying to grasp
      PFlag = False
      startTime = time.time()
      alternateTime = time.time()
      prevTime = 0

      # dataLoggerEnable(False) # start data logging
      # rospy.sleep(0.5) 
      # P_help.stopSampling()
      # rospy.sleep(1)

      # P_help.startSampling()
      # rospy.sleep(0.3)
      # dataLoggerEnable(True)
      # rospy.sleep(1)

      

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
          FT_data = [[FT_help.averageFx_noOffset, FT_help.averageFy_noOffset, FT_help.averageFz_noOffset, FT_help.averageTx_noOffset, FT_help.averageTy_noOffset, FT_help.averageTz_noOffset]]

          # check force limits
          Fx = F_array[0]
          Fy = F_array[1]
          Fz = F_normal
          F_total = np.sqrt(Fx**2 + Fy**2 + Fz**2)

          if F_total > 20:
            # input("net force acting on cup is too high")

            # stop at the last pose
            # rtde_help.stopAtCurrPose()
            rtde_help.stopAtCurrPoseAdaptive()
            rospy.sleep(0.1)
            sequentialFailures+=1
            targetPWM_Pub.publish(DUTYCYCLE_0)
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
            sequentialFailures+=1

            # stop at the last pose
            rtde_help.stopAtCurrPoseAdaptive()
            targetPWM_Pub.publish(DUTYCYCLE_0)
            

            # keep X sec of data after alignment is complete
            rospy.sleep(0.1)
            break
          
          prevTime = time.time()

      #=================== attempt has ended =================== 
      print("time after attempt ends: ", time.time() - pickStartTime)
      args.afterExploreSuccess = suctionSuccessFlag

      #=================== if attempt was successful, pick and place in bin =================== 
      if suctionSuccessFlag: # if the suction finally succeeded, then pick-place operation
        args.transportingSuccess = False
        finalPose = rtde_help.getCurrentPose()

        # option 1 - lift the obejct by setting the suction cup vertically down.
        # LiftPosition = [finalPose.pose.position.x, finalPose.pose.position.y, pick_place_z]
        # PickUpPose = rtde_help.getPoseObj(LiftPosition, setOrientation)

        # # option 2 - lift the object while maintaining the orientation.
        PickUpPose = copy.deepcopy(finalPose)
        PickUpPose.pose.position.z = pick_place_z
        
        # go to pick up pose
        rtde_help.goToPose(PickUpPose)

        # at pick up pose, check if seal was compromised
        P_array = P_help.four_pressure
        pickingUpFailure = 0
        if any(np.array(P_array)>P_vac/2):
          print("seal was compromised while picking up")
          pickingUpFailure = 1
          
          rospy.sleep(0.1)
          suctionSuccessFlag = False
          # turn off vacuum
          targetPWM_Pub.publish(DUTYCYCLE_0)

        # Move to Drop Box
        if suctionSuccessFlag:
          DropPose = copy.deepcopy(dropBoxPose)
          DropPose.pose.position.x = dropBoxPosition[0]
          DropPose.pose.position.y = dropBoxPosition[1]
          DropPose.pose.position.z = dropBoxPosition[2]
          rtde_help.goToPose(DropPose)
          # rospy.sleep(0.1)

        # if seal was compromised while moving to dropbox
        dropBoxFailure = 0
        P_array = P_help.four_pressure
        if any(np.array(P_array)>P_vac/2):
          print("seal was compromised while moving to dropbox")
          dropBoxFailure = 1
          if pickingUpFailure or dropBoxFailure:
            sequentialFailures+=1

          # turn off vacuum
          targetPWM_Pub.publish(DUTYCYCLE_0)
          rospy.sleep(0.02)

        # successful pick and place!
        else:
          args.transportingSuccess = True
          print("seal was not compromised while moving obj to dropbox")
          successfulPicks += 1
          sequentialFailures=0
          ultimateSuccess = 1

        targetPWM_Pub.publish(DUTYCYCLE_0)
        
      else:   # lift up, TAKE A NEW PICTURE AND RERUN
        # input('get final pose')
        finalPose = rtde_help.getCurrentPose()
        finalPose.pose.position.z += 4e-2
        # input('go final pose + some height')  
        rtde_help.goToPose(finalPose)
      
      pickHistory.append(ultimateSuccess)

      # Save Init data
      # input('save input data')
      dataLoggerEnable(False) # start data logging
      args.successfulPicks = successfulPicks
      args.sequentialFailures = sequentialFailures
      args.attemptIdx = attemptIdx
      args.controller_str = controller_str
      args.BrowianMotion = [adpt_help.BM_x, adpt_help.BM_y]
      args.pickHistory = pickHistory
      rospy.sleep(0.5)

      P_help.stopSampling()   
      file_help.saveDataParams(args,appendTxt=fileAppendixStr+'_'+str(attemptIdx) + "_controller_" + controller_str)
      attemptIdx+=1
      print("time to end of attempt: ", time.time() - pickStartTime)
      file_help.clearTmpFolder()
      # save previous points and information
      targetPointFile = open(file_help.ResultSavingDirectory+'/targetPointList.p', 'wb')
      pickle.dump(T_targetPose_prev, targetPointFile)
      targetPointFile.close()
      preTrialInfo = [attemptIdx, successfulPicks, sequentialFailures]
      prevTrialFile = open(file_help.ResultSavingDirectory+'/prevTrialInfo.p', 'wb')
      pickle.dump(preTrialInfo, prevTrialFile)
      prevTrialFile.close()
      args.transportingSuccess = False
      
      

    # print("============ Enter to go to disengage pose ...")
    # input("Enter to go to disengage pose ...")
    
    rtde_help.goToPose(disEngagePose)
    # cartesian_help.goToPose(disEngagePose)
    rospy.sleep(0.1)
    P_help.stopSampling()
    rospy.sleep(1.0)
    targetPWM_Pub.publish(DUTYCYCLE_0)
    rospy.sleep(0.2)

    print("============ Stopping data logger ...")
    print("before dataLoggerEnable(False)")
    print(dataLoggerEnable(False)) # Stop Data Logging
    print("after dataLoggerEnable(False)")
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
  parser.add_argument('--attemptIdx', type=int, help='startIndex Of pick attempt', default= 0)
  parser.add_argument('--successfulPicks', type=int, help='startIndex Of successful picks', default= 0)
  parser.add_argument('--sequentialFailures', type=int, help='startIndex Of failures in a row', default= 0)


  args = parser.parse_args()    
  
  main(args)
  # main(depth, rotAngleList[mode], translateZList[mode])