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

from netft_utils.srv import *
from tae_datalogger.srv import *
from suction_cup.srv import *


def main():


  # controller node
  rospy.init_node('jp_ur_run')

  print("Wait for the PCB segmentation to be enabled")
  rospy.wait_for_service('pcb_segmentation')
  PCB_coord = rospy.ServiceProxy('pcb_segmentation', PCB_location)
  rospy.sleep(1)
  pcb_location = PCB_coord(True) # reset Data Logger just in case
  # print(PCB_location)
  print("pcb_location.x: ", pcb_location.x)
  print("pcb_location.x: ", pcb_location.y)
  print("PCB orientation: ", pcb_location.theta)


if __name__ == '__main__':  
  main()
