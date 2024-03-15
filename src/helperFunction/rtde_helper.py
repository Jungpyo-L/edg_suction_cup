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



class rtdeHelp(object):
    def __init__(self, rtde_frequency = 125, speed = 0.3, acc = 0.2):
        self.tfListener = tf.TransformListener()
        self.rtde_frequency = rtde_frequency

        self.rtde_c = rtde_control.RTDEControlInterface("10.0.0.1", rtde_frequency)
        self.rtde_r = rtde_receive.RTDEReceiveInterface("10.0.0.1", rtde_frequency)

        # rtde_r = RTDEReceive(robot_ip, rtde_frequency, [], True, False, rt_receive_priority)
        # rtde_c = RTDEControl(robot_ip, rtde_frequency, flags, ur_cap_port, rt_control_priority)

        self.checkDistThres = 1e-3
        self.checkQuatThres = 10e-3
        self.transformation = create_transform_matrix(np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]), [0, 0, 0])
        self.speed = speed
        self.acc = acc

    def _append_ns(self, in_ns, suffix):
        """
        Append a sub-namespace (suffix) to the input namespace
        @param in_ns Input namespace
        @type in_ns str
        @return Suffix namespace
        @rtype str
        """
        ns = in_ns
        if ns[-1] != '/':
            ns += '/'
        ns += suffix
        return ns

    def quaternion_multiply(self, q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2,
                w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2)


    def getPoseObj(self, goalPosition, setOrientation):
        Pose = PoseStamped()  
        
        Pose.header.frame_id = "base_link"
        Pose.pose.orientation.x = setOrientation[0]
        Pose.pose.orientation.y = setOrientation[1]
        Pose.pose.orientation.z = setOrientation[2]
        Pose.pose.orientation.w = setOrientation[3]
        
        Pose.pose.position.x = goalPosition[0]
        Pose.pose.position.y = goalPosition[1]
        Pose.pose.position.z = goalPosition[2]
        
        return Pose

    def setCalibrationMatrix(self):
        g1 = adpt_help.get_Tmat_from_Pose(self.getCurrentPoseTF())
        g2 = adpt_help.get_Tmat_from_Pose(self.getCurrentTCPPose())
        # gt = np.matmul(np.linalg.inv(g1), g2)
        gt = np.matmul(g1, np.linalg.inv(g2))
        print("Transformation matrix between pose and TCPpose")
        print(gt)
        self.transformation = gt
    
    def getRotVector(self, goalPose):
        qx = goalPose.pose.orientation.x
        qy = goalPose.pose.orientation.y
        qz = goalPose.pose.orientation.z
        qw = goalPose.pose.orientation.w
        # q = [0, 0, 0, 1] # calculate during the calibration
        # print("goalPose.pose.orientation (converted): ", goalPose.pose.orientation)
        # print("q: ", q)
        # r = R.from_quat(self.quaternion_multiply([qx, qy, qz, qw], q))
        r = R.from_quat([qx, qy, qz, qw])
        Rx, Ry, Rz = r.as_rotvec()
        return Rx, Ry, Rz
    
    def getTransformedPose(self, goalPose):
        T_mat = np.matmul(np.linalg.inv(self.transformation), adpt_help.get_Tmat_from_Pose(goalPose))
        pose = adpt_help.get_ObjectPoseStamped_from_T(T_mat)
        return pose
    
    def getTransformedPoseInv(self, goalPose):
        T_mat = np.matmul(self.transformation, adpt_help.get_Tmat_from_Pose(goalPose))
        pose = adpt_help.get_ObjectPoseStamped_from_T(T_mat)
        return pose

    def getTCPPose(self, pose):
        x = pose.pose.position.x
        y = pose.pose.position.y
        z = pose.pose.position.z
        Rx, Ry, Rz = self.getRotVector(pose)
        return [x, y, z, Rx, Ry, Rz]
    
    def speedl(self, goalPose,speed=0.5, acc=0.5, time=0.5, aRot='a'):

        if len(goalPose) != 6:
            raise ValueError("Target pose must have 6 elements: [x, y, z, Rx, Ry, Rz]")
        try:
        # Perform linear motion using moveL function
            self.rtde_c.speedL(goalPose, self.speed, self.acc, time, aRot)
        except Exception as e:
            print(f"Error occurred during linear motion: {e}")
    
    def getTCPForce(self): # gets (Force/Torque vector) at the TCP
        wrench=self.rtde_r.getActualTCPForce()
        F_x=wrench[0]
        F_y=wrench[1]
        F_z=wrench[2]
        F_m=np.sqrt(F_x**2+F_y**2+F_z**2) #magnitude of forces
        mass=F_m/9.81 #convert to kg
        return [F_x, F_y, F_z, F_m, mass]
    

    def setPayload(self, payload, CoG):
        # Assuming method is avalible within RTDEControlInterface
        self.rtde_c.set_payload(payload, CoG)

    def goToPositionOrientation(self, goalPosition, setOrientation, asynchronous = False):
        self.goToPose(self.getPoseObj(goalPosition, setOrientation))

    # def goToPose(self, goalPose, speed = 2.5, acc = 1.7, asynchronous=False):
    # def goToPose(self, goalPose, speed = 0.1, acc = 0.1, asynchronous=False):     # original? need for edge following
    # def goToPose(self, goalPose, speed = 0.05, acc = 0.01, asynchronous=False):    # try this!
    def goToPose(self, goalPose, speed = 0.3, acc = 1.7, asynchronous=False):        # seb experimenting
    # def goToPose(self, goalPose, speed = 0.25, acc = 0.15, asynchronous=False):          # Alahe
        pose = self.getTransformedPose(goalPose)
        targetPose = self.getTCPPose(pose)
        # speed = self.speed
        # acc = self.acc
        self.rtde_c.moveL(targetPose, speed, acc, asynchronous)
<<<<<<< HEAD
    
    def getTCPForce(self):
        wrench = self.rtde_c.getActualTCPForce()
        F_x = wrench[0]
        F_y = wrench[1]
        F_z = wrench[2]
        F_m = np.sqrt(F_x**2 + F_y**2 + F_z**2)
        mass = F_m/9.81
        return [F_x, F_y, F_z, F_m, mass]
    

    def goToPose2(self, goalPose, speed = 0.0, acc = 0.0, asynchronous=False):
        pose = self.getTransformedPose(goalPose)
        targetPose = self.getTCPPose(pose)
        speed = self.speed
        # acc = self.acc
        self.rtde_c.moveL(targetPose, speed, acc, asynchronous) 
        while not self.checkGoalPoseReached(goalPose):
            distance_threshold=0.1
            if self.checkGoalPoseReached(goalPose, checkDistThres=distance_threshold):
                self.rtde_c.speedL([0,0,0,0,0,0], acc)
=======
        
    def goToPose2(self, goalPose, speed=0.0, acc=0.0, asynchronous=False):
        pose = self.getTransformedPose(goalPose)
        targetPose = self.getTCPPose(pose)
        speed= self.speed
        self.rtde_c.moveL(targetPose, speed, acc, asynchronous)
        while not self.checkGoalPoseReached(goalPose):
            distance_threshold = 0.07
            if self.checkGoalPoseReached(goalPose, checkDistThres=distance_threshold):
                self.rtde_c.speedL([0, 0, 0, 0, 0, 0], acc)  # using speedL to stop once it reached distance threshold
>>>>>>> 49ac72881ac72d1dc21a0ce9056213b64fe0de9d
                break
    # def goToPose(self, goalPose, speed = 0.05, acc = 0.01,  timeCoeff = 10, lookahead_time = 0.1, gain = 200.0):
    #     # lookahead_time range [0.03 0.2]
    #     # grain range [100 2000]
    #     # t_start = self.rtde_c.initPeriod()
    #     pose = self.getTransformedPose(goalPose)
    #     targetPose = self.getTCPPose(pose)
    #     currentPose = self.getTCPPose(self.getCurrentTCPPose())
    #     # print("targetPose-currentPose", np.array(targetPose)-np.array(currentPose))
    #     pose_diff_norm = np.linalg.norm(np.array(targetPose[0:3])-np.array(currentPose[0:3]))
    #     # if pose_diff_norm  > 0.001:
    #     #     print("norm of pose difference: ", pose_diff_norm)

    #     time = pose_diff_norm*timeCoeff
    #     # print("before servoL")
    #     self.rtde_c.servoL(targetPose, speed, acc, time, lookahead_time, gain)

    #     while pose_diff_norm > 0.1e-3:
    #         targetPose = self.getTCPPose(pose)
    #         currentPose = self.getTCPPose(self.getCurrentTCPPose())
    #         pose_diff_norm = np.linalg.norm(np.array(targetPose[0:3])-np.array(currentPose[0:3]))
    #         rospy.sleep(0.01)
    #     # print("norm diff: ", pose_diff_norm)
    #     # self.rtde_c.waitPeriod(t_start)

    def checkGoalPoseReached(self, goalPose, checkDistThres=np.nan, checkQuatThres = np.nan):
        if np.isnan(checkDistThres):
            checkDistThres=self.checkDistThres
        if np.isnan(checkQuatThres):
            checkQuatThres = self.checkQuatThres
        (trans1,rot) = self.tfListener.lookupTransform('/base_link', '/tool0', rospy.Time(0))          
        goalQuat = np.array([goalPose.pose.orientation.x,goalPose.pose.orientation.y, goalPose.pose.orientation.z, goalPose.pose.orientation.w])
        rot_array = np.array(rot)
        quatDiff = np.min([np.max(np.abs(goalQuat - rot_array)), np.max(np.abs(goalQuat + rot_array))])
        distDiff = np.linalg.norm(np.array([goalPose.pose.position.x,goalPose.pose.position.y, goalPose.pose.position.z])- np.array(trans1)) 
        # print(quatDiff, distDiff)
        print("quatdiff: %.4f" % quatDiff)
        print("distDiff: %.4f" % distDiff)
        return distDiff < checkDistThres and quatDiff < checkQuatThres
            
    def goToPoseAdaptive(self, goalPose, speed = 0.0, acc = 0.0,  time = 0.05, lookahead_time = 0.2, gain = 100.0): # normal force measurement
    # def goToPoseAdaptive(self, goalPose, speed = 0.02, acc = 0.02,  time = 0.05, lookahead_time = 0.05, gain = 200.0):
    # def goToPoseAdaptive(self, goalPose, speed = 0.0, acc = 0.0,  time = 0.05, lookahead_time = 0.2, gain = 200.0):
        # lookahead_time range [0.03 0.2]
        # grain range [100 2000]
        t_start = self.rtde_c.initPeriod()
        pose = self.getTransformedPose(goalPose)
        targetPose = self.getTCPPose(pose)
        currentPose = self.getTCPPose(self.getCurrentTCPPose())
        # print("targetPose-currentPose", np.array(targetPose)-np.array(currentPose))
        pose_diff_norm = np.linalg.norm(np.array(targetPose[0:3])-np.array(currentPose[0:3]))
        # if pose_diff_norm  > 0.001:
        #     print("norm of pose difference: ", pose_diff_norm)

        self.rtde_c.servoL(targetPose, speed, acc, time, lookahead_time, gain)
        # rospy.sleep(0.01)
        self.rtde_c.waitPeriod(t_start)

    def goToPoseAdaptive2(self, targetPose, speed = 0.0, acc = 0.0,  time = 0.05, lookahead_time = 0.1, gain = 200.0):
        # lookahead_time range [0.03 0.2]
        # grain range [100 2000]
        t_start = self.rtde_c.initPeriod()
        # pose = self.getTransformedPose(goalPose)
        # targetPose = self.getTCPPose(pose)
        # currentPose = self.getTCPPose(self.getCurrentTCPPose())
        # print("targetPose-currentPose", np.array(targetPose)-np.array(currentPose))
        # pose_diff_norm = np.linalg.norm(np.array(targetPose[0:3])-np.array(currentPose[0:3]))
        # if pose_diff_norm  > 0.001:
        #     print("norm of pose difference: ", pose_diff_norm)

        self.rtde_c.servoL(targetPose, speed, acc, time, lookahead_time, gain)
        # rospy.sleep(0.01)
        self.rtde_c.waitPeriod(t_start)
        
    def readCurrPositionQuat(self):
        (trans1,rot) = self.tfListener.lookupTransform('/base_link', '/tool0', rospy.Time(0))          
        return (trans1, rot) #trans1= position x,y,z, // quaternion: x,y,z,w

    def stopAtCurrPose(self,asynchronous = True):
        currPosition, orientation = self.readCurrPositionQuat()

        # always false
        # wait = False
        self.goToPositionOrientation(currPosition, orientation, asynchronous=asynchronous)
    
    def stopAtCurrPoseAdaptive(self):
        self.rtde_c.servoStop()
        # self.rtde_c.stopScript()

    # Get current pose from TF
    def getCurrentPoseTF(self):
        (Position, Orientation) = self.readCurrPositionQuat()
        return self.getPoseObj(Position, Orientation)
    
    # Get current pose from TCP pose
    def getCurrentPose(self):
        return self.getTransformedPoseInv(self.getCurrentTCPPose())

    def getCurrentPoseTCPaxis(self):
        return self.getTransformedPose(self.getCurrentPose())

    def getCurrentTCPPose(self):
        TCPPose = self.rtde_r.getActualTCPPose()
        Position = [TCPPose[0], TCPPose[1], TCPPose[2]]
        r = R.from_rotvec(np.array([TCPPose[3], TCPPose[4], TCPPose[5]]))
        return self.getPoseObj(Position, r.as_quat())

    def getTCPoffset(self):
        return self.rtde_c.getTCPOffset()

    def setTCPoffset(self, offset):
        return self.rtde_c.setTcp(offset)

    def getMethodsName_r(self):
        object_methods = [method_name for method_name in dir(self.rtde_r) if callable(getattr(self.rtde_r, method_name))]
        print(object_methods)

    def getMethodsName_c(self):
        object_methods = [method_name for method_name in dir(self.rtde_c) if callable(getattr(self.rtde_c, method_name))]
        print(object_methods)