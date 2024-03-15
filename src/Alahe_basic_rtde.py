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
from helperFunction.utils import rotation_from_quaternion, create_transform_matrix, quaternion_from_matrix, normalize, hat


from datetime import datetime
import pandas as pd
import numpy as np
import math
from scipy.io import savemat
from scipy.spatial.transform import Rotation as sciRot


from netft_utils.srv import *
from suction_cup.srv import *
from std_msgs.msg import String
from std_msgs.msg import Int8
import geometry_msgs.msg
import tf

# import rtde_control

from math import pi, cos, sin

from helperFunction.rtde_helper import rtdeHelp
from helperFunction.adaptiveMotion import adaptMotionHelp

def calculate_distance(current_position, target_position):
    return ((current_position[0] - target_position[0])**2 +
            (current_position[1] - target_position[1])**2 +
            (current_position[2] - target_position[2])**2) ** 0.5

def main(args):

  np.set_printoptions(precision=4)

  # controller node
  rospy.init_node('suction_cup')

  # Setup helper functions
  rospy.sleep(0.5)
  rtde_help = rtde_help = rtdeHelp(125)
  rospy.sleep(0.5)
  adpt_help = adaptMotionHelp(dw = 0.5, d_lat = 0.5e-3, d_z = 0.1e-3)

  # Set the TCP offset and calibration matrix
  rospy.sleep(0.5)
  rtde_help.setTCPoffset([0, 0, 0.150, 0, 0, 0])
  rospy.sleep(0.2)
  rtde_help.setCalibrationMatrix()
  rospy.sleep(0.2)

  
  # pose initialization
  disengagePosition_init =  [-0.597, .211, 0.035] # unit is in m
  setOrientation = tf.transformations.quaternion_from_euler(pi,0,pi/2,'sxyz') #static (s) rotating (r), the orientaiton/frame of the trajec
  disEngagePose = rtde_help.getPoseObj(disengagePosition_init, setOrientation) 


  disengagePosition_init2 =  [-0.555, -0.16, 0.035] # unit is in m
  setOrientation = tf.transformations.quaternion_from_euler(pi,0,pi/2,'sxyz') #static (s) rotating (r), the orientaiton/frame of the trajec
  disEngagePose2 = rtde_help.getPoseObj(disengagePosition_init2, setOrientation) 

  
  print("disEngagePose: ",  disEngagePose.pose.position)
  target = [disEngagePose.pose.position.x,  disEngagePose.pose.position.y,  disEngagePose.pose.position.z]
  # target = [disEngagePose2.pose.position.x,  disEngagePose2.pose.position.y,  disEngagePose2.pose.position.z]
  # currentPose = rtde_help.getPoseObj(rtde_help.getCurrentTCPPose())
  currentPose = rtde_help.getCurrentPose()
  current = [currentPose.pose.position.x,  currentPose.pose.position.y,  currentPose.pose.position.z]

  distance = calculate_distance(current, target)
  # distance2 = calculate_distance(current, target2)
  #subscribing to receive pose active
  # data = rtdeHelp.receive()
  # #looking for the real time pose
  # rPose = data.actual_tool_pose
  current1=[0,0,0]
  # target=[0,0,0]

  # try block so that we can have a keyboard exception
  try:
    # Go to disengage Pose
    input("Press <Enter> to go disEngagePose")
    rtde_help.goToPose(disEngagePose)
    print(rtde_help.getCurrentPose())
    
    # rtde_help.goToPoseAdaptive(disEngagePose)
    rospy.sleep(0.5)
    #displaying the distance at each step 
    
    if not distance <0.01: 
      # current_pose1 = rtde_help.getTCPPose(disEngagePose) #attempt to get updated pose as the UR10 is moving
      current_pose1 = rtde_help.getCurrentPose() #attempt to get updated pose as the UR10 is moving
      current1 = [current_pose1.pose.position.x,  current_pose1.pose.position.y,  current_pose1.pose.position.z]
      # target1 = [disEngagePose2.pose.position.x,  disEngagePose2.pose.position.y,  disEngagePose2.pose.position.z]

      # target1 = [disEngagePose.pose.position.x,  disEngagePose.pose.position.y,  disEngagePose.pose.position.z]
      print("initial pose: ", current)
      print("Final pose: ", current1)
      print("Target: ", target)
      print("calc distance w final pose: ", calculate_distance(current1, target))

    # distance2= disEngagePose-rtde_help.getTCPPose(disEngagePose)
    print("Calc distance w init pose: ", distance)
    # print(rtde_help.getActualTCPForce())
    # print(rtde_help.getActualTCPPose())
    print("============ Python UR_Interface demo complete!")

    # rtde_help.goToPose(disEngagePose)




    
  
  except rospy.ROSInterruptException:
    return
  except KeyboardInterrupt:
    return  


if __name__ == '__main__':  
  import argparse
  parser = argparse.ArgumentParser()
  args = parser.parse_args()    
  
  main(args)
