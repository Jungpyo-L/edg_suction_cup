#!/usr/bin/env python

import os
import datetime
import numpy as np
import re
from geometry_msgs.msg import PoseStamped
from .utils import rotation_from_quaternion, create_transform_matrix, quaternion_from_matrix, normalize, hat
from scipy.spatial.transform import Rotation as Rot
import scipy  
#!/usr/bin/env python
try:
  import rospy
  import tf
  ros_enabled = True
except:
  print('Couldn\'t import ROS.  I assume you\'re running this on your laptop')
  ros_enabled = False

from grp import getgrall
from hmac import trans_36
import numpy as np
from geometry_msgs.msg import PoseStamped

from helperFunction.adaptiveMotion import adaptMotionHelp
from .utils import create_transform_matrix

import rtde_control
import rtde_receive

from tf.transformations import quaternion_matrix

from scipy.spatial.transform import Rotation as R
import copy
import numpy as np
adpt_help = adaptMotionHelp()



def get_ObjectPoseStamped_from_T(self,T):
    thisPose = PoseStamped()
    thisPose.header.frame_id = "base_link"
    R = T[0:3,0:3]
    # quat = quaternion_from_matrix(R)
    quat =R
    position = T[0:3,3]
    [thisPose.pose.position.x, thisPose.pose.position.y, thisPose.pose.position.z] = position
    [thisPose.pose.orientation.x, thisPose.pose.orientation.y, thisPose.pose.orientation.z,thisPose.pose.orientation.w] = quat

    return thisPose

def get_Tmat_from_Pose(self,PoseStamped):  #*
        quat = [PoseStamped.pose.orientation.x, PoseStamped.pose.orientation.y, PoseStamped.pose.orientation.z, PoseStamped.pose.orientation.w]        
        translate = [PoseStamped.pose.position.x, PoseStamped.pose.position.y, PoseStamped.pose.position.z]
        return self.get_Tmat_from_PositionQuat(translate, quat)
    
def get_Tmat_from_PositionQuat(self, Position, Quat):    #*
        # rotationMat = rotation_from_quaternion(Quat)
        rotationMat=Quat
        T = create_transform_matrix(rotationMat, Position)
        return T


def get_PoseStamped_from_T_initPose(self, T, initPoseStamped):   #*
    T_now = self.get_Tmat_from_Pose(initPoseStamped)
    # targetPose = self.get_ObjectPoseStamped_from_T(np.matmul(T_now, T))
    targetPose = self.get_ObjectPoseStamped_from_T(np.matmul(np.linalg.inv(T_now), T))
    return targetPose

def get_Tmat_TranlateInBodyF(self, translate = [0., 0., 0.]): #*
    return create_transform_matrix(np.eye(3), translate)

def get_Tmat_TranlateInZ(self, direction = 1):     #*
    offset = [0.0, 0.0, np.sign(direction)*self.d_z_normal]
    # if step:
    #     offset = [0.0, 0.0, np.sign(direction)*step]
    return self.get_Tmat_TranlateInBodyF(translate = offset)

def setCalibrationMatrix(self):
    g1 = adpt_help.get_Tmat_from_Pose(self.getCurrentPoseTF())
    g2 = adpt_help.get_Tmat_from_Pose(self.getCurrentTCPPose())
    gt = np.matmul(np.linalg.inv(g1), g2)
    # gt = np.matmul(g1, np.linalg.inv(g2))
    print("Transformation matrix between pose and TCPpose")
    print(gt)
    self.transformation = gt