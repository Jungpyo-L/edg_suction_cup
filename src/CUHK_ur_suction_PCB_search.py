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
import tf

from math import pi, cos, sin

from std_msgs.msg import String
from std_msgs.msg import Int8
from geometry_msgs.msg import WrenchStamped
from geometry_msgs.msg import PoseStamped
import geometry_msgs.msg

from netft_utils.srv import *
from tae_datalogger.srv import *
from suction_cup.srv import *

from helperFunction.SuctionP_callback_helper import P_CallbackHelp
from helperFunction.FT_callback_helper import FT_CallbackHelp
from helperFunction.fileSaveHelper import fileSaveHelp
from helperFunction.rtde_helper import rtdeHelp
from helperFunction.adaptiveMotion import adaptMotionHelp
from helperFunction.gqcnn_policy_class import GraspProcessor

import pyrealsense2 as rs
from perception import RgbdSensorFactory

def discover_cams():
    """Returns a list of the ids of all cameras connected via USB."""
    ctx = rs.context()
    ctx_devs = list(ctx.query_devices())
    ids = []
    for i in range(ctx.devices.size()):
        ids.append(ctx_devs[i].get_info(rs.camera_info.serial_number))
    return ids

def getRealSenseIntr():
## =========== Read Camera Img =============== ##
    ids = discover_cams()
    assert ids, "[!] No camera detected."

    cfg = {}
    cfg["cam_id"] = ids[0]
    cfg["filter_depth"] = True
    cfg["frame"] = "realsense_overhead"

    sensor = RgbdSensorFactory.sensor("realsense", cfg)
    sensor.start()
    time.sleep(1) # needs some time to restart the camera.
    camera_intr = sensor.color_intrinsics
    return camera_intr

def main(args):

  deg2rad = np.pi / 180.0
  DUTYCYCLE_100 = 100
  DUTYCYCLE_30 = 30
  DUTYCYCLE_0 = 0
  rtde_frequency = 125.0
  F_normalThres = [1.5, 2.0]

  AxisAngleThres = np.pi/4.0
  disToPrevThres = 20e-3
  timeLimit = 15.0 # searching time limit  
  dispLimit = 50e-3 # displacement limit
  args.timeLimit = timeLimit
  args.dispLimit = dispLimit
  args.timeOverFlag = False
  args.dispOverFlag = False
  args.forceLimitFlag = False

  # controller node
  rospy.init_node('jp_ur_run')

  # Setup helper functions
  FT_help = FT_CallbackHelp() # it deals with subscription.
  rospy.sleep(0.5)
  P_help = P_CallbackHelp() # it deals with subscription.
  rospy.sleep(0.5)
  rtde_help = rtdeHelp(rtde_frequency)
  rospy.sleep(0.5)
  file_help = fileSaveHelp()
  # adpt_help = adaptMotionHelp()
  adpt_help = adaptMotionHelp(dw = 0.5, d_lat = args.dLateral, d_z = 0.1e-3)

  # Set the PWM Publisher  
  targetPWM_Pub = rospy.Publisher('pwm', Int8, queue_size=1)
  targetPWM_Pub.publish(DUTYCYCLE_0)

  # Set data logger
  print("Wait for the data_logger to be enabled")
  rospy.wait_for_service('data_logging')
  dataLoggerEnable = rospy.ServiceProxy('data_logging', Enable)
  dataLoggerEnable(False) # reset Data Logger just in case
  rospy.sleep(1)
  file_help.clearTmpFolder()        # clear the temporary folder

  # Set PCB segmentation
  print("Wait for the PCB segmentation to be enabled")
  rospy.wait_for_service('pcb_segmentation')
  PCB_coord = rospy.ServiceProxy('pcb_segmentation', PCB_location)
  rospy.sleep(1)

  # Load Camera Transform matrix    
  datadir = os.path.dirname(os.path.realpath(__file__))
  with open(datadir + '/TransformMat_board_verified', 'rb') as handle:
      loaded_data = pickle.load(handle)
  T_cam = loaded_data
  print("T_cam: ", T_cam)
  
  # Load Scale factor
  with open(datadir + '/scaleFactor.p', 'rb') as handle:
    scale_Factor = pickle.load(handle)
  print("Scale factor: ", scale_Factor)

  # Load initial pose
  with open(datadir + '/initEndEffectorPose.p', 'rb') as handle:
    initEndEffectorPose = pickle.load(handle)
  print("camera pose from calibration: ", initEndEffectorPose)

  # get intrinsic matrix of camera
  camera_intr = getRealSenseIntr()

  # pose initialization
  disengagePosition =  [-690e-3, 85e-3, 420e-3]
  setOrientation = tf.transformations.quaternion_from_euler(pi,0,pi/2,'sxyz')
  disEngagePose = rtde_help.getPoseObj(disengagePosition, setOrientation)
  scale_offset = disengagePosition[2]-initEndEffectorPose.position.z
  scale_Final = scale_Factor + scale_offset - args.height_comp # height compensating factor

  rtde_help.setTCPoffset([0, 0, 0.150, 0, 0, 0])
  rtde_help.setCalibrationMatrix()
  # print("calibration matrix", rtde_help.transformation)

  input("Press <Enter> to go disEngagePose")
  print("Start to go to disEngagePose")
  rtde_help.goToPose(disEngagePose)
  # print("disEngagePOse: ", disEngagePose)
  rospy.sleep(0.5)
  initEndEffPoseStamped = rtde_help.getCurrentPose()

  targetPWM_Pub.publish(DUTYCYCLE_100)
  rospy.sleep(0.1)

  # start sampling pressure and FT data, bias both
  P_help.startSampling()
  rospy.sleep(0.3)
  
  # set biases now
  try:
    FT_help.setNowAsBias()
    rospy.sleep(0.1)
    P_help.setNowAsOffset()
    rospy.sleep(0.1)
  except:
    print("set now as offset failed, but it's okay")

  
  try:
    while True:
      args.timeOverFlag = False
      args.dispOverFlag = False
      args.forceLimitFlag = False
      print("input the name of PCB (1-8) when it is ready")
      x = input()
      args.PCB = str(x)

      # input("Press <Enter> to go disEngagePose")
      print("Start to go to disEngagePose")
      rtde_help.goToPose(disEngagePose)
      # print("disEngagePOse: ", disEngagePose)
      rospy.sleep(0.1)
      initEndEffPoseStamped = rtde_help.getCurrentPose()
      targetPWM_Pub.publish(DUTYCYCLE_100)

      # input("Press <Enter> to take a photo and get a PCB pose")
      print("Start to take a photo")
      pcb_location = PCB_coord(True)
      print("pcb_location.x: ", pcb_location.x)
      print("pcb_location.x: ", pcb_location.y)
      print("PCB orientation: ", pcb_location.theta)
      pixel_frame = np.array([pcb_location.x, pcb_location.y, 1])
      PCB_center = np.matmul(np.linalg.inv(camera_intr.K), pixel_frame)
      PCB_Position_cam = scale_Final*PCB_center
      targetPoseStamped = adpt_help.getGoalPosestampedFromCam(T_cam, PCB_Position_cam, initEndEffPoseStamped)
      # print("targetPoseStampled: ", targetPoseStamped)

      # start data logger
      P_help.startSampling()
      rospy.sleep(0.1)
      dataLoggerEnable(True) # start data logging
      suctionSuccessFlag = False

      # input("Press <Enter> to approach to targetPose")
      print("Start to approach to target Pose")
      T_offset = adpt_help.get_Tmat_TranlateInBodyF([0., 0., -15e-3]) # small offset from the target pose
      targetSearchPoseStamped = adpt_help.get_PoseStamped_from_T_initPose(T_offset, targetPoseStamped)
      rtde_help.goToPose(targetSearchPoseStamped)
      rospy.sleep(0.1)


      # input("press <Enter> to start haptic search")
      print("Start to haptic search")
      # flag for alternating, time that controller starts trying to grasp
      PFlag = False
      startTime = time.time()
      alternateTime = time.time()
      prevTime = 0

      # get initial pose to check limits
      T_N_Engaged = adpt_help.get_Tmat_from_Pose(targetPoseStamped) # relative angle limit
      T_Engaged_N = np.linalg.inv(T_N_Engaged)
      
      while not suctionSuccessFlag:
        # 1. down to target pose
        rtde_help.goToPose(targetPoseStamped, speed=args.speed, acc=args.acc)
        rospy.sleep(args.waitTime)

        # calculate axis angle
        measuredCurrPose = rtde_help.getCurrentPose()
        T_N_curr = adpt_help.get_Tmat_from_Pose(measuredCurrPose)          
        T_Engaged_curr = T_Engaged_N @ T_N_curr

        # calculate current pos/angle
        displacement = np.linalg.norm(T_Engaged_curr[0:3,3])

        # 2. get sensor data
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

        if F_total > 12:
          print("net force acting on cup is too high")
          args.forceLimitFlag = True
          # stop at the last pose
          rtde_help.stopAtCurrPose()
          rospy.sleep(0.1)
          break
        
        # 3. lift up and check suction success
        targetSearchPoseStamped = adpt_help.get_PoseStamped_from_T_initPose(T_offset, targetPoseStamped)
        rtde_help.goToPose(targetSearchPoseStamped, speed=args.speed, acc=args.acc)
        
        # P_vac = P_help.P_vac
        P_vac = -2500.0
        if int(args.PCB) == 7:
          P_vac = -1500.0
        elif int(args.PCB) == 8:
          P_vac = -1000.0        
        P_array_check = P_help.four_pressure
        reached_vacuum = all(np.array(P_array_check)<P_vac)

        if reached_vacuum:
          # vacuum seal formed, success!
          suctionSuccessFlag = True
          print("Suction engage succeeded with controller")

          # stop at the last pose
          rtde_help.stopAtCurrPoseAdaptive()
          args.elapedTime = time.time()-startTime
          break

        elif time.time()-startTime > timeLimit or displacement > dispLimit:
          args.timeOverFlag = time.time()-startTime >timeLimit
          args.dispOverFlag = displacement > dispLimit

          suctionSuccessFlag = False
          print("Haptic search fail")
          # stop at the last pose
          rtde_help.stopAtCurrPoseAdaptive()
          targetPWM_Pub.publish(DUTYCYCLE_0)
          rospy.sleep(0.1)
          break

        # 4. move to above the next search location
        T_later = adpt_help.get_Tmat_lateralMove(P_array)
        targetPoseStamped = adpt_help.get_PoseStamped_from_T_initPose(T_later, measuredCurrPose)
        targetSearchPoseStamped = adpt_help.get_PoseStamped_from_T_initPose(T_offset, targetPoseStamped)
        rtde_help.goToPose(targetSearchPoseStamped, speed=args.speed, acc=args.acc)

      
      # Save Init data
      dataLoggerEnable(False) # start data logging
      args.suctionSuccessFlag = suctionSuccessFlag
      file_help.saveDataParams(args, appendTxt='jp_haptic_jumping_search'+'_PCB_'+str(args.PCB)+'_step_' + str(args.dLateral)+'_acc_' + str(args.acc)+'_waitTime_' + str(args.waitTime))
      file_help.clearTmpFolder()
      P_help.stopSampling()
      rospy.sleep(0.5)


      # input("Press <Enter> to go disEngagePose")
      print("Go to disEngagePose")
      rtde_help.goToPose(disEngagePose)
      rospy.sleep(0.5)
      targetPWM_Pub.publish(DUTYCYCLE_0)


    print("============ Python UR_Interface demo complete!")


  except rospy.ROSInterruptException:
    return
  except KeyboardInterrupt:
    return


if __name__ == '__main__':
  
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--speed', type=float, help='take image or use existingFile', default=0.3)
  parser.add_argument('--acc', type=float, help='location of target saved File', default=0.6)
  parser.add_argument('--waitTime', type=float, help='location of target saved File', default=0.1)
  parser.add_argument('--dLateral', type=float, help='location of target saved File', default=0.005)
  parser.add_argument('--height_comp', type=float, help='height compensate factor for scale', default=0.002)
  parser.add_argument('--PCB', type=str, help='label of test PCB from 0 to 1', default='PCB')
  args = parser.parse_args()    
  main(args)
