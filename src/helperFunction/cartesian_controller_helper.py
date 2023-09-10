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

from controller_manager_msgs.msg import ControllerState
from controller_manager_msgs.srv import *
from controller_manager_msgs.utils\
    import ControllerLister, ControllerManagerLister,\
    get_rosparam_controller_names
from dynamic_reconfigure.srv import *
from dynamic_reconfigure.msg import Config
import copy





class pdParam:
  def __init__(self, input): 
    self.name = input[0]
    self.value = input[1]


class cartCtrlHelp(object):
    def __init__(self):
        self.targetPose_Pub = rospy.Publisher('my_cartesian_motion_controller/target_frame', PoseStamped, queue_size=1)
        self.tfListener = tf.TransformListener()
        self.dealwithController(mode = 0)
        self.registeredPose = []
        self.checkDistThres = 1e-3
        self.checkQuatThres = 10e-3

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


    def dealwithController(self, mode = 0):    
        list_cm = ControllerManagerLister()
        print(list_cm()[0])
        cm_ns = list_cm()[0]
        switch_srv_name = self._append_ns(cm_ns, 'switch_controller')
        switch_srv = rospy.ServiceProxy(switch_srv_name, SwitchController, persistent=True)
        
        controllers = ControllerLister(list_cm()[0])()
        
        allControllerNames = []
        for controller in controllers:
            allControllerNames.append(controller.name)
        allControllerNames = tuple(allControllerNames)  
        
        bestEffort = SwitchControllerRequest.BEST_EFFORT
        
        req = SwitchControllerRequest(start_controllers=[],stop_controllers=allControllerNames,strictness=bestEffort)    
        switch_srv.call(req)
        rospy.sleep(.01)

        if mode == 0:    
            onController = ('my_cartesian_motion_controller', 'joint_state_controller')
            req = SwitchControllerRequest(start_controllers=onController,stop_controllers=[],strictness=bestEffort)
            switch_srv.call(req)
            rospy.sleep(.01)
        elif mode == 1: # Turn on the handle.
            onController = ('my_cartesian_motion_controller', 'joint_state_controller','my_motion_control_handle')
            req = SwitchControllerRequest(start_controllers=onController,stop_controllers=[],strictness=bestEffort)
            switch_srv.call(req)
            rospy.sleep(.01)

    def setGainVals(self, inputScale=0.5):  
        if inputScale > 1:
            inputScale = 1
        elif inputScale < 0:
            inputScale = 0
        
        rotScale = 1.0
        tempStruct = Config()
        tempStruct2 = Config()
        Pgain = 10.0
        Dgain = 10.0
        # Pgain = 10.0
        # Dgain = 10.0
        tempStruct.doubles = [pdParam(['p',Pgain*inputScale]),pdParam(['d',Dgain*inputScale]) ]
        tempStruct2.doubles = [pdParam(['p',Pgain*inputScale*rotScale]),pdParam(['d',Dgain*inputScale*rotScale]) ]
        # tempStruct.doubles = [pdParam(['p',6.0*inputScale]),pdParam(['d',0.01*inputScale]) ]
        # tempStruct.doubles = [pdParam(['p',8.5*inputScale]),pdParam(['d',2*inputScale]) ]
        # tempStruct.doubles = [pdParam(['p',Pgain*inputScale]),pdParam(['d',Dgain*inputScale]) ]
        # tempStruct2.doubles = [pdParam(['p',Pgain*inputScale*rotScale]),pdParam(['d',Dgain*inputScale*rotScale]) ]

        transX = rospy.ServiceProxy('my_cartesian_motion_controller/pd_gains/trans_x/set_parameters', Reconfigure)  
        transX.call(tempStruct)
        transY = rospy.ServiceProxy('my_cartesian_motion_controller/pd_gains/trans_y/set_parameters', Reconfigure)  
        transY.call(tempStruct)
        transZ = rospy.ServiceProxy('my_cartesian_motion_controller/pd_gains/trans_z/set_parameters', Reconfigure)  
        transZ.call(tempStruct)
        
        rotX = rospy.ServiceProxy('my_cartesian_motion_controller/pd_gains/rot_x/set_parameters', Reconfigure)  
        rotX.call(tempStruct2)
        rotY = rospy.ServiceProxy('my_cartesian_motion_controller/pd_gains/rot_y/set_parameters', Reconfigure)  
        rotY.call(tempStruct2)
        rotZ = rospy.ServiceProxy('my_cartesian_motion_controller/pd_gains/rot_z/set_parameters', Reconfigure)  
        rotZ.call(tempStruct2)

        rospy.sleep(0.7)  


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
    
    def registerPose(self, goalPosition, setOrientation):        
        self.registeredPose.append(getPoseObj(goalPosition, setOrientation))        
        registeredId = len(self.registeredPose)-1
        return registeredId

    def goToPositionOrientation(self, goalPosition, setOrientation, wait = False):
        self.goToPose(self.getPoseObj(goalPosition, setOrientation),wait=wait)

    def goToPose(self, goalPose, wait=False):
        self.targetPose_Pub.publish(goalPose)   
        if wait:
            self.waitForGoalPose(goalPose)

    def goToPoseID(self, goalId,wait = False):
        goalPose = self.registeredPose[goalId]
        self.targetPose_Pub.publish(goalPose)
        if wait:
            self.waitForGoalPose(goalPose)

    def waitForGoalPose(self, goalPose):
        while True:
            if self.checkGoalPoseReached(goalPose):            
                break

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
        print("quatDiff: ",quatDiff)
        # print("rot[2,2]: ", rot[2,2])
        # print("rot: ", rot)
        return distDiff < checkDistThres and quatDiff < checkQuatThres
            

    def readCurrPositionQuat(self):
        (trans1,rot) = self.tfListener.lookupTransform('/base_link', '/tool0', rospy.Time(0))          
        return (trans1, rot) #trans1= position x,y,z, // quaternion: x,y,z,w

    def stopAtCurrPose(self,wait = False):
        currPosition, orientation = self.readCurrPositionQuat()

        # always false
        # wait = False
        self.goToPositionOrientation(currPosition, orientation, wait=wait)

    def getCurrentPose(self):
        (Position, Orientation) = self.readCurrPositionQuat()
        return self.getPoseObj(Position, Orientation)
    
    def getWaypointNum(self, goalPoseStamped):
        currentPose = self.getCurrentPose()

        v1 = np.zeros((3, ), dtype=np.float64)
        v1[0] = currentPose.pose.position.x
        v1[1] = currentPose.pose.position.y
        v1[2] = currentPose.pose.position.z

        v2 = np.zeros((3, ), dtype=np.float64)
        v2[0] = goalPoseStamped.pose.position.x
        v2[1] = goalPoseStamped.pose.position.y
        v2[2] = goalPoseStamped.pose.position.z

        # iterNum = int(np.linalg.norm(v2-v1)*50) # 1 way point per 2 cm
        # iterNum = int(np.linalg.norm(v2-v1)*100) # 1 way point per 1 cm
        iterNum = int(np.linalg.norm(v2-v1)*80)
        if iterNum == 0:
            iterNum = 1

        return iterNum

    def goToPoseGradually(self, goalPoseStamped, iterNum = 10, wait=True, controlFreq = 3.0):
        # iterNum = self.getWaypointNum(goalPoseStamped)
        waypoints = self.getGradualWaypointsFromCurrent(goalPoseStamped=goalPoseStamped, iterateNum=iterNum)
        print("iterNum in goTOPoseGradually: ", iterNum)
        rate = rospy.Rate(controlFreq)
        i = 0
        # wait = False
        for targetPoseStamped in waypoints:
            self.goToPose(targetPoseStamped, wait=False)
            # print(i)
            i+=1
            rate.sleep()        
        if wait:
            # rospy.sleep(0.2)
            self.goToPose(waypoints[-1], wait=True)


    def getGradualWaypointsFromCurrent(self, goalPoseStamped, iterateNum=5):        
        currentPose = self.getCurrentPose()        
        
        v1 = np.zeros((3, ), dtype=np.float64)
        q1 = np.zeros((4, ), dtype=np.float64)
        v1[0] = currentPose.pose.position.x
        v1[1] = currentPose.pose.position.y
        v1[2] = currentPose.pose.position.z
        q1[0] = currentPose.pose.orientation.x
        q1[1] = currentPose.pose.orientation.y
        q1[2] = currentPose.pose.orientation.z
        q1[3] = currentPose.pose.orientation.w

        v2 = np.zeros((3, ), dtype=np.float64)
        q2 = np.zeros((4, ), dtype=np.float64)
        v2[0] = goalPoseStamped.pose.position.x
        v2[1] = goalPoseStamped.pose.position.y
        v2[2] = goalPoseStamped.pose.position.z
        q2[0] = goalPoseStamped.pose.orientation.x
        q2[1] = goalPoseStamped.pose.orientation.y
        q2[2] = goalPoseStamped.pose.orientation.z
        q2[3] = goalPoseStamped.pose.orientation.w

        tempPose = PoseStamped()  
        tempPose.header.frame_id = "base_link"

        waypoints = []
        for idx in range(0,iterateNum+1):
            fraction = idx / iterateNum
            q_temp = tf.transformations.quaternion_slerp(q1, q2, fraction)
            v_temp = v1 + (v2 - v1) * fraction
            
            tempPose.pose.position.x = v_temp[0]
            tempPose.pose.position.y = v_temp[1]
            tempPose.pose.position.z = v_temp[2]
            tempPose.pose.orientation.x = q_temp[0]
            tempPose.pose.orientation.y = q_temp[1]
            tempPose.pose.orientation.z = q_temp[2]
            tempPose.pose.orientation.w = q_temp[3]

            waypoints.append(copy.deepcopy(tempPose))

        return waypoints # new robot Pose
