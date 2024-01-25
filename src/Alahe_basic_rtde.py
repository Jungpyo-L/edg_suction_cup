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
  disengagePosition_init =  [-0.597, .211, 0.025] # unit is in m
  setOrientation = tf.transformations.quaternion_from_euler(pi,0,pi/2,'sxyz') #static (s) rotating (r), the orientaiton/frame of the trajec
  disEngagePose = rtde_help.getPoseObj(disengagePosition_init, setOrientation) 

  def calculate_distance(current_position, target_position):
    return ((current_position[0] - target_position[0])**2 +
            (current_position[1] - target_position[1])**2 +
            (current_position[2] - target_position[2])**2) ** 0.5
  
  distance = calculate_distance(disengagePosition_init, disEngagePose)

  #subscribing to receive pose actively
  # data = rtdeHelp.receive()
  # #looking for the real time pose
  # rPose = data.actual_tool_pose


  # try block so that we can have a keyboard exception
  try:
    # Go to disengage Pose
    input("Press <Enter> to go disEngagePose")
    rtde_help.goToPose(disEngagePose)  #goes to the pose, in attempt to make a second attempt creates roundoff error
    # rtde_help.goToPoseAdaptive(disEngagePose)
    rospy.sleep(0.1)
    #displaying the distance at each step 
    
    while not distance <0.01: 
      current_pose = rtde_help.getTCPPose(disEngagePose) #attempt to get updated pose as the UR10 is moving
      print("Current pose is: ", current_pose)
      print("Received distance is: ", calculate_distance(current_pose, disengagePosition_init))

    
    print("Calcualted distance is: ", distance)
    print("============ Python UR_Interface demo complete!")

    
  
  except rospy.ROSInterruptException:
    return
  except KeyboardInterrupt:
    return  


if __name__ == '__main__':  
  import argparse
  parser = argparse.ArgumentParser()
  args = parser.parse_args()    
  
  main(args)
