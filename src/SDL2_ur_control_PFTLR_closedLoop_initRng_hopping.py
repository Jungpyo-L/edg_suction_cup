#!/usr/bin/env python

# Authors: Sebastian D. Lee and Tae Myung Huh and Jungpyo Lee

# test change

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
import matplotlib.pyplot as plt
import random


from moveGroupInterface_Tae import MoveGroupInterface
from scipy.io import savemat
from scipy.spatial.transform import Rotation as sciRot

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
# from transitions import Machine

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
from scipy import linalg

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
import argparse


class URControl:
    def __init__(self):
        parser = argparse.ArgumentParser()
        self.args = parser.parse_args()

        self.initialize_params()

    def initialize_params(self):
        
        self.thetaIdx = 0

        # PARAMETER SWEEP (theta, offset, yaw)
        # self.thetaList = -np.array(range(0, 46, 5)) / 180 * np.pi
        self.theta = 0
        self.args.offset = 0
        self.args.phi = 0
        # self.args.phi = 45 / 180 * np.pi


        # CHOOSE THE FEATURE
        # flat-edge-tilt
        self.engagePosition = [-(500e-3 - 058e-3), 200e-3 + 049e-3, 24e-3]
        self.args.domeRadius = 9999
        self.args.edge = 1
        self.disengageOffset = 2e-3

        # # tilted flat-edge-tilt
        # self.engagePosition[0] += 000e-3
        # self.engagePosition[1] -= 100e-3
        # self.args.edge = -20

        # # tilted flat-edge-tilt
        # self.engagePosition[0] += 000e-3
        # self.engagePosition[1] -= 140e-3
        # self.args.edge = 20
        # self.disengageOffset = 6e-3

        # # dome-tilt
        # self.engagePosition[0] += 000e-3
        # self.engagePosition[1] += 040e-3
        # self.args.domeRadius = 20
        # self.args.edge = 0

        # # # flat-tilt
        # self.engagePosition[0] -= 040e-3
        # self.engagePosition[1] -= 040e-3
        # self.args.domeRadius = 9999
        # self.args.edge = 0
        

        # CONSTANT TRANSFORMS

        # set offset for the given starting point
        self.engagePosition[0] += self.args.offset / 1000

        # disengage position
        self.disengagePosition = self.engagePosition.copy()
        self.disengagePosition[2] += self.disengageOffset

        # normal force
        self.F_normalThres = 1.5
        self.F_lim = 1.5
        self.Fz_tolerance = 0.1

        # frequencies
        self.rtde_frequency = 125
        self.DUTYCYCLE_100 = 100
        self.DUTYCYCLE_30 = 30
        self.DUTYCYCLE_0 = 0

        # cup constants
        self.r_cup = 11.5e-3 + 2e-3
        self.omega_hat1 = hat(np.array([1, 0, 0]))
        self.omega_hat2 = hat(np.array([0, 0, 1]))

        self.angleLimit = np.pi / 4


    def initialize(self):
        try:
            rospy.init_node('tae_ur_run')
            self.FT_help = FT_CallbackHelp()
            rospy.sleep(0.1)
            self.P_help = P_CallbackHelp()
            rospy.sleep(0.1)
            self.rtde_help = rtdeHelp(self.rtde_frequency)
            rospy.sleep(0.1)
            self.file_help = fileSaveHelp()
            rospy.sleep(0.1)
            self.adpt_help = adaptMotionHelp(dw=0.5, d_lat=0.5e-3, d_z=0.2e-3)
            rospy.sleep(0.1)

        except Exception as e:
            self.log_error("Initialization", e)
            raise

    def set_tcp_and_calibration(self):
        try:
            self.rtde_help.setTCPoffset([0, 0, 0.156, 0, 0, 0])
            rospy.sleep(0.1)
        except Exception as e:
            self.log_error("Set TCP and Calibration", e)
            raise

    def set_initial_pose(self):
        try:
            # make cup face down
            self.setOrientation = tf.transformations.quaternion_from_euler(pi, 0, -pi / 2 - pi, 'sxyz')
            self.engagePose = self.rtde_help.getPoseObj(self.engagePosition, self.setOrientation)
            self.disEngagePose = self.rtde_help.getPoseObj(self.disengagePosition, self.setOrientation)
            rospy.sleep(0.1)
            self.rtde_help.goToPose(self.engagePose)
            rospy.sleep(0.1)
            
        except Exception as e:
            self.log_error("Set Initial Pose", e)
            raise

    def wait_for_user_input(self):
        input("Press Enter to continue...")

    def start_data_logger(self):
        try:
            print("Wait for the data_logger to be enabled")
            rospy.wait_for_service('data_logging')
            self.dataLoggerEnable = rospy.ServiceProxy('data_logging', Enable)
            self.dataLoggerEnable(False) # reset Data Logger just in case
            rospy.sleep(1.0)
            self.file_help.clearTmpFolder()
        except:
            self.log_error("Start data logger", e)
            raise


    def move_to_engage_pose(self):
        try:
            self.targetPWM_Pub = rospy.Publisher('pwm', Int8, queue_size=1)
            rospy.sleep(0.1)
            self.targetPWM_Pub.publish(self.DUTYCYCLE_30)
            rospy.sleep(0.1)
            self.rtde_help.goToPose(self.disEngagePose)
            rospy.sleep(0.1)
            self.rtde_help.goToPose(self.engagePose)
            rospy.sleep(0.1)
        except Exception as e:
            self.log_error("Move to Engage Pose", e)
            raise

    def start_sampling(self):
        try:
            self.P_help.startSampling()
            rospy.sleep(0.1)
        except Exception as e:
            self.log_error("Start Sampling", e)
            raise

    def set_bias(self):
        try:
            self.biasNotSet = True
            while self.biasNotSet:
                try:
                    self.FT_help.setNowAsBias()
                    rospy.sleep(0.05)
                    self.P_help.setNowAsOffset()
                    rospy.sleep(0.05)
                    self.biasNotSet = False
                except Exception as e:
                    print("set now as offset failed, but it's okay")

        except Exception as e:
            self.log_error("Set Bias", e)
            raise

    def adjust_tip(self):
        try:
            self.tipContactPose = self.rtde_help.getCurrentPose()
            self.tipContactPose.pose.position.z -= 0e-3
            self.rtde_help.goToPose(self.tipContactPose)
            rospy.sleep(0.1)
            
        except Exception as e:
            self.log_error("Adjust tip", e)
            raise

    def start_sampling_and_logging(self):
        self.P_help.startSampling()
        rospy.sleep(0.5)
        self.dataLoggerEnable(True)

    # def move_to_initial_positions(self, theta):
    #     # rotation matrices
    #     self.theta = theta
    #     self.Rw1 = linalg.expm(self.theta * self.omega_hat1)
    #     self.Rw2 = linalg.expm(self.args.phi * self.omega_hat2)
    #     self.Rw = np.dot(self.Rw1, self.Rw2)

    #     # define point behind grasp point
    #     self.L = self.r_cup * np.sin(np.abs(self.theta)) + self.disengageOffset
    #     self.cx = self.L * np.sin(self.theta)
    #     self.cz = -self.L * np.cos(np.abs(self.theta))

    #     # create transformation matrix
    #     self.T_from_tipContact = create_transform_matrix(self.Rw, [0.0, self.cx, self.cz])
    #     self.targetPose = self.adpt_help.get_PoseStamped_from_T_initPose(self.T_from_tipContact, self.tipContactPose)
        
    #     # go to the target pose
    #     self.rtde_help.goToPose(self.targetPose)
    #     rospy.sleep(0.5)

    #     # pwm stamp for 1.5 seconds
    #     self.targetPWM_Pub.publish(self.DUTYCYCLE_100)
    #     rospy.sleep(1.5)
    #     self.targetPWM_Pub.publish(self.DUTYCYCLE_0)
        
    #     # self.T_from_tipContact = create_transform_matrix(self.Rw, [0.0, self.cx, self.cz])
    #     # self.targetPose = self.adpt_help.get_PoseStamped_from_T_initPose(self.T_from_tipContact, self.tipContactPose)
    #     # self.rtde_help.goToPose(self.targetPose)
    #     # rospy.sleep(0.5)
    #     # self.targetPWM_Pub.publish(self.DUTYCYCLE_100)
    #     # rospy.sleep(1.5)
    #     # self.targetPWM_Pub.publish(self.DUTYCYCLE_0)
    
    def move_to_initRng_positions(self, theta, offset):
        # rotation matrices
        self.theta = theta
        self.Rw1 = linalg.expm(self.theta * self.omega_hat1)
        self.Rw2 = linalg.expm(self.args.phi * self.omega_hat2)
        self.Rw = np.dot(self.Rw1, self.Rw2)

        # define point behind grasp point
        self.L = self.r_cup * np.sin(np.abs(self.theta)) + self.disengageOffset
        self.cx = self.L * np.sin(self.theta) + offset / 1000
        self.cz = -self.L * np.cos(np.abs(self.theta))

        # create transformation matrix
        self.T_from_tipContact = create_transform_matrix(self.Rw, [0.0, self.cx, self.cz])
        self.targetPose = self.adpt_help.get_PoseStamped_from_T_initPose(self.T_from_tipContact, self.tipContactPose)
        
        # go to the target pose
        self.rtde_help.goToPose(self.targetPose)
        rospy.sleep(0.5)

        # pwm stamp for 1.5 seconds
        self.targetPWM_Pub.publish(self.DUTYCYCLE_100)
        rospy.sleep(1.5)
        self.targetPWM_Pub.publish(self.DUTYCYCLE_0)

    def check_force_and_pressure(self):
        self.P_array = self.P_help.four_pressure
        self.P_curr = np.mean(self.P_help.four_pressure)
        self.F_normal = self.FT_help.averageFz_noOffset
        self.dF = self.F_normal - (-self.F_normalThres)
        self.Fy = self.FT_help.averageFy_noOffset

        if np.abs(self.dF) < self.Fz_tolerance or self.P_curr < self.P_vac:
            self.inRangeCounter += 1
        else:
            self.inRangeCounter = 0
        print("fz counter:", self.inRangeCounter)

    def adjust_pose(self):
        if self.F_normal > -(self.F_normalThres - self.Fz_tolerance):
            T_normalMove = self.adpt_help.get_Tmat_TranlateInZ(direction=1)
        elif self.F_normal < -(self.F_normalThres + self.Fz_tolerance):
            T_normalMove = self.adpt_help.get_Tmat_TranlateInZ(direction=-1)
        else:
            T_normalMove = np.eye(4)
        T_move = T_normalMove

        self.currentPose = self.rtde_help.getCurrentPose()
        self.targetPose_adjusted = self.adpt_help.get_PoseStamped_from_T_initPose(T_move, self.currentPose)
        self.rtde_help.goToPoseAdaptive(self.targetPose_adjusted)
    
    def adjust_pose_adaptively(self):
        try:
            T_align = np.eye(4)
            T_later = np.eye(4)
            weightVal = 2

            if self.F_normal > -(self.F_normalThres - self.Fz_tolerance):
                T_normalMove = self.adpt_help.get_Tmat_TranlateInZ(direction=1)
            elif self.F_normal < -(self.F_normalThres + self.Fz_tolerance):
                T_normalMove = self.adpt_help.get_Tmat_TranlateInZ(direction=-1)
            else:
                T_normalMove = np.eye(4)
                T_align, T_later, weightVal = self.adpt_help.get_Tmats_dpFxy(self.P_array, self.Fy)
            # T_move = T_normalMove

            self.weightVal = weightVal

            # HERE I CAN INJECT ROTATION AND TRANSLATION
            # T_normalMove = self.adpt_help.get_Tmat_axialMove(self.F_normal, self.F_normalThres)
            # T_align, T_later = self.adpt_help.get_Tmats_Suction(weightVal=0.0)
            # T_align, T_later = self.adpt_help.get_Tmats_alignSuctionLateralMode(self.P_array, weightVal=1.0)
            

            # ic(self.P_array)
            # ic(T_align)
            # ic(T_later)

            T_move =  T_later @ T_align @ T_normalMove

            self.currentPose = self.rtde_help.getCurrentPose()
            self.targetPose_adjusted = self.adpt_help.get_PoseStamped_from_T_initPose(T_move, self.currentPose)
            self.rtde_help.goToPoseAdaptive(self.targetPose_adjusted)
        except Exception as e:
            self.log_error("Adjust Pose Adaptively", e)
            raise

    def save_data_and_increment(self):
        try:
            # if self.inRangeCounter > 100:   # change this condition to be time-based and pressure-based
            # ic(self.P_curr)
            # ic(self.P_vac)
            # ic(self.previous_time)

            reached_vacuum = abs(self.P_curr) > abs(self.P_vac)
            attempt_time = time.time() - self.previous_time

            # ic(attempt_time)
            # ic(reached_vacuum)
            measuredCurrPose = self.rtde_help.getCurrentPose()
            T_N_curr = self.adpt_help.get_Tmat_from_Pose(measuredCurrPose)          
            T_Engaged_curr = self.T_Engaged_N @ T_N_curr
            angleDiff = np.arccos(T_Engaged_curr[2,2])

            if reached_vacuum or attempt_time > 60 or angleDiff > self.angleLimit:
                self.rtde_help.stopAtCurrPoseAdaptive()
                rospy.sleep(0.1)
                self.targetPWM_Pub.publish(self.DUTYCYCLE_0)
                rospy.sleep(0.5)
                print("Theta:", self.theta, "reached with Fz", self.F_normal)
                self.targetPWM_Pub.publish(self.DUTYCYCLE_100)

                self.thetaIdx += 1
                self.thisThetaNeverVisited = True

                rospy.sleep(1.5)
                self.targetPWM_Pub.publish(self.DUTYCYCLE_0)
                rospy.sleep(0.2)
                self.dataLoggerEnable(False)
                rospy.sleep(0.2)

                self.args.Fz_set = self.F_normalThres
                self.args.gamma = int(round(self.theta * 180 / pi))
                self.args.phi = int(round(self.args.phi * 180 / pi))
                self.file_help.saveDataParams(self.args, appendTxt='seb_rotational_' + 'domeRadius' + str(self.args.domeRadius) + 'mm_gamma' + str(self.args.gamma) + '_phi' + str(self.args.phi) + '_edge' + str(self.args.edge) + '_offset' + str(self.args.offset))
                self.file_help.clearTmpFolder()
                self.P_help.stopSampling()
                rospy.sleep(0.1)

            print("time since start of grasp attempt:", attempt_time)
            print("Fy: ", self.Fy)
            print("weightVal: ", self.weightVal)
            # print("target theta: ", self.theta / pi * 180)
            print("P_curr: ", self.P_curr)
            print("dF: ", self.dF)
            # print("np.abs(dF)<Fz_tolerance: ", np.abs(self.dF) < self.Fz_tolerance)

            # ic(angleDiff)
            print("")
        except Exception as e:
            self.log_error("Save Data and Increment", e)
            raise

    # def sweep_loop(self):
    #     try:
    #         self.targetPWM_Pub.publish(self.DUTYCYCLE_30)
    #         self.thisThetaNeverVisited = True
    #         self.P_vac = self.P_help.P_vac
    #         self.P_curr = np.mean(self.P_help.four_pressure)
    #         while self.thetaIdx < len(self.thetaList):
    #             if self.thisThetaNeverVisited:
    #                 self.start_sampling_and_logging()
    #                 self.move_to_initial_positions(self.thetaList[self.thetaIdx])
    #                 self.thisThetaNeverVisited = False
    #                 self.inRangeCounter = 0
    #                 rospy.sleep(0.5)
    #                 self.targetPWM_Pub.publish(self.DUTYCYCLE_100)

    #             self.check_force_and_pressure()
    #             self.adjust_pose()
    #             self.save_data_and_increment()
    #     except Exception as e:
    #         self.log_error("Sweep Loop", e)
    #         raise
    
    def grasp_attempt_initRng(self):
        try:
            self.targetPWM_Pub.publish(self.DUTYCYCLE_30)
            self.thisThetaNeverVisited = True
            self.P_vac = self.P_help.P_vac
            self.P_array = self.P_help.four_pressure
            self.P_curr = np.mean(self.P_help.four_pressure)
            self.previous_time = time.time()
            self.thetaIdx = 0

            targetPoseEngaged = self.rtde_help.getCurrentPose()
            T_N_Engaged = self.adpt_help.get_Tmat_from_Pose(targetPoseEngaged) # relative angle limit
            self.T_Engaged_N = np.linalg.inv(T_N_Engaged)

            while self.thetaIdx < 1:
                if self.thisThetaNeverVisited:
                    self.start_sampling_and_logging()
                    self.move_to_initRng_positions(self.args.theta, self.args.offset)
                    self.thisThetaNeverVisited = False
                    self.inRangeCounter = 0
                    rospy.sleep(0.5)
                    self.targetPWM_Pub.publish(self.DUTYCYCLE_100)

                self.check_force_and_pressure()
                self.adjust_pose_adaptively()
                self.save_data_and_increment()
            
        except Exception as e:
            self.log_error("RNG grasp attempt", e)
            raise
            

    def stop_sampling(self):
        try:
            self.rtde_help.stopAtCurrPoseAdaptive()
            rospy.sleep(0.5)
            self.dataLoggerEnable(False)
            rospy.sleep(0.5)
            self.P_help.stopSampling()
            self.targetPWM_Pub.publish(self.DUTYCYCLE_0)
            rospy.sleep(0.5)
        except Exception as e:
            self.log_error("Stop Sampling", e)
            raise

    def move_to_disengage_pose(self):
        try:
            self.setOrientation = tf.transformations.quaternion_from_euler(pi, 0, -pi / 2 - pi, 'sxyz')
            self.disEngagePose = self.rtde_help.getPoseObj(self.disengagePosition, self.setOrientation)
            rospy.sleep(1)
            self.rtde_help.goToPose(self.disEngagePose)
            rospy.sleep(1)
            print("============ Python UR_Interface demo complete!")
        except Exception as e:
            self.log_error("Save Data", e)
            raise

    def handle_error(self):
        print("An error occurred during the state machine execution.")
        # Handle cleanup and recovery here

    def log_error(self, context, exception):
        print(f"Error in {context}: {exception}")

# if __name__ == '__main__':
#     control = URControl()
#     control.next()

if __name__ == '__main__':
    control = URControl()

    # Example of calling functions directly
    try:
        control.initialize()  # changes per experiment
        control.set_tcp_and_calibration()
        control.set_initial_pose()
        control.start_data_logger()
        control.move_to_engage_pose()
        control.start_sampling()
        control.set_bias()
        control.adjust_tip()

        # control.sweep_loop()

        # if randomized, this could be loop for a # of iterations
        for i in range(1):
            control.args.theta = 20 / 180 *pi
            control.args.offset = -3
            # control.args.theta = random.randint(-25, 35) / 180 * pi
            # control.args.offset = random.randint(-12, 14) * 0.5
            control.grasp_attempt_initRng()

        control.stop_sampling()
        control.move_to_disengage_pose()

        print("Execution complete!")
    except Exception as e:
        control.log_error("Main Execution", e)