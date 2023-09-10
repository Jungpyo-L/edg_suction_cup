#!/usr/bin/env python

from mimetypes import init
import os, sys
import rospy
from moveGroupInterface_Tae import MoveGroupInterface
from scipy.io import savemat
from datetime import datetime
import pandas as pd
import re
import subprocess
import numpy as np
import copy

import cv2
from cv2 import aruco

from tae_psoc.msg import cmdToPsoc
from edg_data_logger.srv import *
from std_msgs.msg import String

import matplotlib as mpl
mpl.use("TkAgg")
# mpl.use("Agg")
from autolab_core import Logger, Point
import matplotlib.pyplot as plt
import pyrealsense2 as rs
import time
# from perception.perception.rgbd_sensors import RgbdSensorFactory
from perception import RgbdSensorFactory
from visualization import Visualizer2D as vis2d

import pickle
from helperFunction.rigid_transform_3D.rigid_transform_3D import rigid_transform_3D
from helperFunction.fileSaveHelper import fileSaveHelp


def discover_cams():
    """Returns a list of the ids of all cameras connected via USB."""
    ctx = rs.context()
    ctx_devs = list(ctx.query_devices())
    ids = []
    for i in range(ctx.devices.size()):
        ids.append(ctx_devs[i].get_info(rs.camera_info.serial_number))
    return ids

def findArucoMarkers(img, markerSize=4, totalMarkers=50, draw=True):
      # """
      # :param img: iage in which to find the aruco markers
      # :param markerSize: the size of the markers
      # :param totalMarkers: total number of markers that composethe dictionary
      # :param draw: flag to draw bbox around markers detected
      # :return: bounding boes and id numbers of markers detected
      # """
      
      imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      key = getattr(aruco,'DICT_4X4_50') # For now let's force the marker key.
      # key = getattr(aruco,f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
      arucoDict = aruco.Dictionary_get(key)
      arucoParam = aruco.DetectorParameters_create()
      bboxs, ids, rejected = aruco.detectMarkers(imgGray, arucoDict, parameters=arucoParam)
      #print(ids)
      
      if draw:
          aruco.drawDetectedMarkers(img, bboxs)
          
      return bboxs, ids

def getRealSenseFrame():
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
    
    # print(camera_intr)
    # print("intrinsics matrix: {}".format(camera_intr.K))
    color_im, depth_im   = sensor.frames()
    sensor.stop()

    print("intrinsics matrix: {}".format(camera_intr.K))

    # _, axes = plt.subplots(1, 2)
    # for ax, im in zip(axes, [color_im.data, depth_im.data]):
    #     ax.imshow(im)
    #     ax.axis("off")

    return color_im, depth_im, camera_intr
      
def checkForceOutput():
    # First Check all CSV files in /tmp/ and bring them in as a variable  
  fileList = []
  for file in os.listdir("/tmp"):    
    if file.endswith(".csv") and file.find("netft"):
        fileList.append(os.path.join("/tmp", file))
        
  thisFT_CSV = sorted(fileList)[-1]
  df=pd.read_csv(thisFT_CSV)
  dataArray = df.values

  xDiffer = np.amax(dataArray[:,1]) - np.amin(dataArray[:,1])
  yDiffer = np.amax(dataArray[:,2]) - np.amin(dataArray[:,2])
  zDiffer = np.amax(dataArray[:,3]) - np.amin(dataArray[:,3])

  print("Force Diff in x, y, z")
  print(xDiffer)
  print(yDiffer)
  print(zDiffer)


def getT_from_R_t(RotMat, t_vec):
    T = np.concatenate((RotMat, t_vec),axis=1)
    T = np.concatenate((T, np.array([[0.,0.,0.,1.]])))
    return T


def main():
  try:
    print("Position Test")
    rospy.init_node('jp_ur_run')

 
    
    print("")
    print("----------------------------------------------------------")
    print("Welcome to the MoveIt MoveGroup Python Interface UR_Interface")
    print("----------------------------------------------------------")
    print("Press Ctrl-D to exit at any time")
    print("")
    print("============ Begin the tutorial by setting up the moveit_commander ...")
    UR_Interface = MoveGroupInterface()
    file_help = fileSaveHelp()

    
    # engagePosition_Z0 = [-0.59, -.09, 0.26]# When depth is 0 cm. unit is in m
    engagePosition_Z0 = [-0.72, .03, 0.3]# When depth is 0 cm. unit is in m (for new bin picking configuration (230125))
    UR_Interface.engaging_endEffector_Position = engagePosition_Z0


    print("============ Press `Enter` to execute a Engage Pose")
    input()    
    if not UR_Interface.go_to_engagePose():
          UR_Interface.go_to_engagePose()
    rospy.sleep(0.5)
    if not UR_Interface.go_to_engagePose():
          UR_Interface.go_to_engagePose()
    rospy.sleep(0.5)
    initEndEffectorPose = copy.deepcopy(UR_Interface.move_group.get_current_pose().pose)

    print("============ Press `Enter` to Get Image")
    input()    
    color_im, depth_im, camera_intr= getRealSenseFrame()
    intrinsic_matrix = camera_intr.K 


    ############# Tracking Each Individual Marker
    bboxs,ids = findArucoMarkers(color_im.data)
    
    centerPixels = []
    centerPoints = []
    
    vis2d.figure(size=(5, 5))
    vis2d.imshow(color_im)
    # vis2d.figure(size=(5, 5))
    # vis2d.imshow(self.depth_im, vmin=0.15, vmax=0.35)

    color = plt.get_cmap('hsv')(0.3)[:-1]

    for bbox in bboxs:
      thisCenter=np.mean(bbox,axis = 1, dtype=np.int32)[0]
      centerPixels.append(thisCenter)      
      centerPoints.append(camera_intr.deproject_pixel(depth_im[thisCenter[1]][thisCenter[0]], Point(np.array(thisCenter), frame=camera_intr.frame)))
      plt.scatter(*thisCenter, color=color, marker=".", s=100)
    
    centerPoints_cam_sorted = []
    centerPixels_cam_sorted = []
    idOrder = np.argsort(ids,axis=0)
    for orderIdx in idOrder:
      centerPoints_cam_sorted.append(centerPoints[orderIdx[0]])
      centerPixels_cam_sorted.append(centerPixels[orderIdx[0]])

    plt.show(block=False)


    #build 3xN coordinate matrix    
    centerCoords_in_cam = np.empty((3,0))
    for centerPoint in centerPoints_cam_sorted:
      # print("centerPoint: ", centerPoint)
      thisCoord = np.array([[centerPoint.x],[centerPoint.y],[centerPoint.z]])
      centerCoords_in_cam = np.concatenate((centerCoords_in_cam, thisCoord), axis=1)

    centerPixels_in_cam = np.empty((3,0))
    centerCoords_in_cam_from_pixel = np.empty((3,0))
    for centerPoint in centerPixels_cam_sorted:
      # print("centerPoint: ", centerPoint)
      thisCoord = np.array([[centerPoint[0]],[centerPoint[1]],[1]])
      thisCoord_cam = np.matmul(np.linalg.inv(intrinsic_matrix), thisCoord)
      centerPixels_in_cam = np.concatenate((centerPixels_in_cam, thisCoord), axis=1)
      centerCoords_in_cam_from_pixel = np.concatenate((centerCoords_in_cam_from_pixel, thisCoord_cam), axis=1)

    scale_factor = np.mean(centerCoords_in_cam[2]/centerCoords_in_cam_from_pixel[2])
    print("scale_factor: ", scale_factor)
    # print("centerCoords_in_cam_from_pixel: ", centerCoords_in_cam_from_pixel)
    # print("ratio 1: ", centerCoords_in_cam[0]/centerCoords_in_cam_from_pixel[0])
    # print("ratio 1: ", centerCoords_in_cam[1]/centerCoords_in_cam_from_pixel[1])
    # print("ratio 1: ", centerCoords_in_cam[2]/centerCoords_in_cam_from_pixel[2])

  
    #=========== Collect the Four Corner Points ================
    robotTabPoints = []
    print("============ Press `Enter` After reaching Point 0 (Top Left)")        
    input()
    print(UR_Interface.move_group.get_current_pose().pose)
    robotTabPoints.append(copy.deepcopy(UR_Interface.move_group.get_current_pose().pose))
    
    print("============ Press `Enter` After reaching Point 1 (Top Right)")        
    input()
    print(UR_Interface.move_group.get_current_pose().pose)
    robotTabPoints.append(copy.deepcopy(UR_Interface.move_group.get_current_pose().pose))

    print("============ Press `Enter` After reaching Point 2 (Bottom Left)")        
    input()
    print(UR_Interface.move_group.get_current_pose().pose)
    robotTabPoints.append(copy.deepcopy(UR_Interface.move_group.get_current_pose().pose))
    
    print("============ Press `Enter` After reaching Point 3 (Bottom Right)")        
    input()
    print(UR_Interface.move_group.get_current_pose().pose)
    robotTabPoints.append(copy.deepcopy(UR_Interface.move_group.get_current_pose().pose))

    # Get 4 corner center coordinates
    z_fixed = 9e-3 #From Experiments
    fourCenterCoords_in_Robot = np.empty((3,0))    
    for robotPoint in robotTabPoints:
      x = robotPoint.position.x-initEndEffectorPose.position.x
      y = robotPoint.position.y-initEndEffectorPose.position.y
      z = z_fixed-initEndEffectorPose.position.z
      thisCoord = np.array([[x],[y],[z]])
      fourCenterCoords_in_Robot = np.concatenate((fourCenterCoords_in_Robot, thisCoord), axis=1)


    # Coordinates in Board frame.
    markerWidth = 25e-3
    markerGap = 10e-3
    markerXNum = 5
    markerYNum = 7
    

    centerCoord_in_board = np.zeros((3, markerXNum*markerYNum))    
    markerIdx = 0
    for yi in range(markerYNum):
      for xi in range(markerXNum):
        centerCoord_in_board[:,markerIdx] = np.array([-xi*(markerWidth+markerGap),yi*(markerWidth+markerGap),0.0])
        markerIdx+=1

    fourCenterCoords_in_board = centerCoord_in_board[:,[0,markerXNum-1,markerXNum*(markerYNum-1),markerXNum*markerYNum-1]]


    # Compute Transform matrix
    # B = R@A + t
    # Recover R and t
    # ret_R, ret_t = rigid_transform_3D(A, B) # R_B_A, t_B_A
    
    # T_robot_board
    ret_R, ret_t = rigid_transform_3D(fourCenterCoords_in_board, fourCenterCoords_in_Robot)
    T_robot_board = getT_from_R_t(ret_R, ret_t)

    # T_board_cam
    ret_R, ret_t = rigid_transform_3D(centerCoords_in_cam, centerCoord_in_board)
    T_board_cam = getT_from_R_t(ret_R, ret_t)

    T_robot_cam = T_robot_board @ T_board_cam

    T = T_robot_cam


    robotInitPoseArray = np.array([initEndEffectorPose.position.x, initEndEffectorPose.position.y, initEndEffectorPose.position.z])

    EstimatedRobotPositions = []
    for imagePoint in centerPoints_cam_sorted:
      deltaPosition = np.matmul( T, (np.array([imagePoint.x, imagePoint.y, imagePoint.z, 1])))
      EstimatedRobotPositions.append( deltaPosition[0:3] + robotInitPoseArray)

    testIdxList = [0, 4, 20, 34]
    for testIdx in testIdxList:        
      print(EstimatedRobotPositions[testIdx])
      print("============ Press `Enter` to Marker " + str(testIdx))
      input()
      UR_Interface.go_to_engagePose()
      UR_Interface.go_to_Position(EstimatedRobotPositions[testIdx] + np.array([0., 0., 5e-3]),speedScale=0.2,wantWait = True)
      rospy.sleep(0.5)
      UR_Interface.go_to_Position(EstimatedRobotPositions[testIdx] + np.array([0., 0., 5e-3]),speedScale=0.2,wantWait = True)


    print("============ Press `Enter` if this Transformation is verified")

    input()
    currentFileDirect = os.path.dirname(os.path.abspath(__file__))

    file = open(currentFileDirect+'/TransformMat_board_verified', 'wb')
    pickle.dump(T, file)
    file.close()
    scaleFactorFile = open(currentFileDirect+'/scaleFactor.p', 'wb')
    pickle.dump(scale_factor, scaleFactorFile)
    scaleFactorFile.close()
    endEffectorPoseFile = open(currentFileDirect+'/initEndEffectorPose.p', 'wb')
    pickle.dump(initEndEffectorPose, endEffectorPoseFile)
    endEffectorPoseFile.close()
    

    print("============ Press `Enter` to execute DisengagePose")
    input()
    print(UR_Interface.go_to_disengagePose_simple())


    print("============ Python UR_Interface demo complete!")
  except rospy.ROSInterruptException:
    return
  except KeyboardInterrupt:
    return

if __name__ == '__main__':  
  main()


