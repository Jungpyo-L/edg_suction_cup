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
  disengagePosition_init =  [-0.597, .211, 0.025] # unit is in m
  setOrientation = tf.transformations.quaternion_from_euler(pi,0,pi/2,'sxyz') #static (s) rotating (r), the orientaiton/frame of the trajec
  disEngagePose = rtde_help.getPoseObj(disengagePosition_init, setOrientation) 
  currentPose = rtde_help.getCurrentTCPPose()

  current = [currentPose.pose.position.x, currentPose.pose.position.y, currentPose.pose.position.z]

  disengagePosition_init2 =  [-0.557, .151, 0.025] # unit is in m
  setOrientation = tf.transformations.quaternion_from_euler(pi,0,pi/2,'sxyz') #static (s) rotating (r), the orientaiton/frame of the trajec
  disEngagePose2 = rtde_help.getPoseObj(disengagePosition_init2, setOrientation) 

  
  print("disEngagePose: ",  disEngagePose.pose.position)
  target = [disEngagePose.pose.position.x,  disEngagePose.pose.position.y,  disEngagePose.pose.position.z]
  
  distance=0
  distance = calculate_distance(current, target)
  # distance2 = disEngagePose-currentPose


  #subscribing to receive pose active
  # data = rtdeHelp.receive()
  # #looking for the real time pose
  # rPose = data.actual_tool_pose
  current1=[0,0,0]
  target1=[0,0,0]

  # try block so that we can have a keyboard exception
  try:
    # Go to disengage Pose
    input("Press <Enter> to go disEngagePose")
    # rtde_help.goToPose(disEngagePose)
    rtde_help.goToPoseAdaptive(disEngagePose)
    print(rtde_help.getCurrentPose())
    
    # rtde_help.goToPoseAdaptive(disEngagePose)
    rospy.sleep(0.5)
    #displaying the distance at each step 
    
    if not distance <0.01: 
      current_pose = rtde_help.getCurrentTCPPose() #attempt to get updated pose as the UR10 is moving
      current1 = [current_pose.pose.position.x, current_pose.pose.position.y, current_pose.pose.position.z]
      target1 = [disEngagePose.pose.position.x,  disEngagePose.pose.position.y,  disEngagePose.pose.position.z]
      print("Current pose is: ", current1)
      print("calculated distance: ", calculate_distance(current1, target1)) 

    
    print("initial distance: ", distance) #distance2 being the difference between isEngagePose and currentPose
    #print(rtde_help.getTCPForce())
    # modifying payload incase getTCPForce is not zero or close to zero
    # payload_mass = 2.0  
    # payload_CoG = [0.0, 0.0, 200]  
    # rtde_help.setPayload(payload_mass, payload_CoG)
    # rtde_help.goToPose2(disEngagePose)
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
