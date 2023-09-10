import sys
import copy
import rospy
import moveit_commander

import moveit_msgs.msg
import geometry_msgs.msg
from math import pi, cos, sin
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list

import tf
import numpy
import itertools
import random

import endeffectorOffset as eff_offsetCal



## END_SUB_TUTORIAL


def all_close(goal, actual, tolerance):
  """
  Convenience method for testing if a list of values are within a tolerance of their counterparts in another list
  @param: goal       A list of floats, a Pose or a PoseStamped
  @param: actual     A list of floats, a Pose or a PoseStamped
  @param: tolerance  A float
  @returns: bool
  """
  all_equal = True
  if type(goal) is list:
    #For cartesian coordinate
    for index in range(0,3):
      if abs(actual[index] - goal[index]) > tolerance:
        return False
    #For Quaternion
    for index in range(3,len(goal)):
      if abs(actual[index] - goal[index]) > tolerance and abs(actual[index] - (-goal[index])) > tolerance:        
        return False

  elif type(goal) is geometry_msgs.msg.PoseStamped:
    return all_close(goal.pose, actual.pose, tolerance)

  elif type(goal) is geometry_msgs.msg.Pose:    
    return all_close(pose_to_list(goal), pose_to_list(actual), tolerance)

  return True


class MoveGroupInterface(object):
  """MoveGroupPythonIntefaceTutorial"""
  
  initOrientaionSXYZ = [pi, 0, pi/2]
  # initOrientaionSXYZ = [pi, 0, 0] # for the new bin picking orientation (jp: 230125)

  def __init__(self, seedNum=0):
    super(MoveGroupInterface, self).__init__()
    
    #global variable in this Class
    self.endEffector_offset = eff_offsetCal.endEffector_offset #defined in "eff_offsetCal.py"
    self.engaging_endEffector_Position = [-0.460, .346, 0.2772] #[-0.450, .346, 0.325]
    # self.engaging_endEffector_Position = [-0.425, .346, 0.301]

    
    ## BEGIN_SUB_TUTORIAL setup
    ##
    ## First initialize `moveit_commander`_ and a `rospy`_ node:
    moveit_commander.roscpp_initialize(sys.argv)
    # rospy.init_node('tae_ur_run', anonymous=True)

    ## Instantiate a `RobotCommander`_ object. Provides information such as the robot's
    ## kinematic model and the robot's current joint states
    robot = moveit_commander.RobotCommander()

    ## Instantiate a `PlanningSceneInterface`_ object.  This provides a remote interface
    ## for getting, setting, and updating the robot's internal understanding of the
    ## surrounding world:
    scene = moveit_commander.PlanningSceneInterface()

    
   
    
    ## Instantiate a `MoveGroupCommander`_ object.  This object is an interface
    ## to a planning group (group of joints).  In this tutorial the group is the primary
    ## arm joints in the Panda robot, so we set the group's name to "panda_arm".
    ## If you are using a different robot, change this value to the name of your robot
    ## arm planning group.
    ## This interface can be used to plan and execute motions:
    #group_name = "panda_arm"
    group_name = "manipulator"
    
    move_group = moveit_commander.MoveGroupCommander(group_name)

    
    ## Create a `DisplayTrajectory`_ ROS publisher which is used to display
    ## trajectories in Rviz:
    display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                   moveit_msgs.msg.DisplayTrajectory,
                                                   queue_size=20)
    

    ## END_SUB_TUTORIAL

    ## BEGIN_SUB_TUTORIAL basic_info
    ##
    ## Getting Basic Information
    ## ^^^^^^^^^^^^^^^^^^^^^^^^^
    # We can get the name of the reference frame for this robot:
    planning_frame = move_group.get_planning_frame()
    print("============ Planning frame: %s" % planning_frame)
    
    poseRefFrame = move_group.get_pose_reference_frame()
    print("============ Pose ref frame: %s" % poseRefFrame)

    # We can also print the name of the end-effector link for this group:
    move_group.set_end_effector_link("tool0")
    eef_link = move_group.get_end_effector_link()
    print("============ End effector link: %s" % eef_link)
    
    # We can get a list of all the groups in the robot:
    group_names = robot.get_group_names()
    # print "============ Available Planning Groups:", robot.get_group_names()

    # Sometimes for debugging it is useful to print the entire state of the
    # robot:
    # print "============ Printing robot state"
    # print robot.get_current_state()
    print("")
    ## END_SUB_TUTORIAL

    # Generate motion lists
    self.genrateMotionList(seedNum)

    # Misc variables
    self.box_name = ''
    self.robot = robot
    self.scene = scene
    self.move_group = move_group
    self.display_trajectory_publisher = display_trajectory_publisher
    

    self.planning_frame = planning_frame
    self.eef_link = eef_link
    self.group_names = group_names

    # Clean the view
    # scene.remove_world_object()
    self.wait_for_state_update()
    
  # def robot2suctioncup(self):
    
  def genrateMotionList(self, seedNum):    
    #spherical coordinate
    # thetas = numpy.linspace(pi, pi*3/2, 4) # angle from x
    thetas = numpy.linspace(0, 2*pi, 13) # angle from x, 30deg
    thetas = thetas[0:-1]
    phis = numpy.linspace(0, pi, 9)   # angle from z
    translates_xy = [0.0]
    translate_z = [0.02]
    self.motionList = list(itertools.product(thetas, phis, translates_xy, translates_xy, translate_z))

    if seedNum != 0:
      # Add random variation
      tmpArray = numpy.asarray(self.motionList)
      # Angle Random    
      numpy.random.seed(seedNum)
      deltaTheta = thetas[1]-thetas[0]
      deltaPhi = phis[1]-phis[0]
      deltaXYZ = 0.01 # For translation, max random is fixed to 1cm.

      randTheta = numpy.random.uniform(-deltaTheta/2,deltaTheta/2, (tmpArray.shape[0],1))
      randPhi = numpy.random.uniform(-deltaPhi/2,deltaPhi/2, (tmpArray.shape[0],1))
      randXYZ = numpy.random.uniform(-deltaXYZ,deltaXYZ, (tmpArray.shape[0],3))

      randArray = numpy.concatenate((randTheta,randPhi,randXYZ), axis=1)
      newRandomizedArray = tmpArray + randArray

      self.motionList = newRandomizedArray.tolist()
      # end of randomizing

    #randomize the motion List with seed
    random.seed(seedNum)
    random.shuffle(self.motionList)




  def go_to_initial_joint_state(self):
    # Copy class variables to local variables to make the web tutorials more clear.
    # In practice, you should use the class variables directly unless you have a good
    # reason not to.
    move_group = self.move_group

    

    ## BEGIN_SUB_TUTORIAL plan_to_joint_state
    ##
    ## Planning to a Joint Goal
    ## ^^^^^^^^^^^^^^^^^^^^^^^^
    ## The Panda's zero configuration is at a `singularity <https://www.quora.com/Robotics-What-is-meant-by-kinematic-singularity>`_ so the first
    ## thing we want to do is move it to a slightly better configuration.
    # We can get the joint values from the group and adjust some of the values:
    joint_goal = move_group.get_current_joint_values()
    # joint_goal[0] = 0
    # joint_goal[1] = -pi/4
    # joint_goal[2] = 0
    # joint_goal[3] = -pi/2
    # joint_goal[4] = 0
    # joint_goal[5] = pi/3
    
    # This values should be re adjusted given the robot's configuration.
    # joint_goal = [-1.7940612912430232, 4.593054650904408, -1.8808965250884422, -1.1050456646159932, 1.5789800073056186, 2.4676931189145197]
    joint_goal = [-3.194594208394186, -1.7566359678851526, 1.8939275741577148, -1.6666811148272913, -1.5822213331805628, -0.3011415640460413]

    move_group.set_max_velocity_scaling_factor(0.5)
    move_group.set_max_acceleration_scaling_factor(0.5)
    
    # The go command can be called with joint values, poses, or without any
    # parameters if you have already set the pose or joint target for the group
    move_group.go(joint_goal, wait=True)
    # move_group.go(joint_goal, wait=True)

    rospy.sleep(0.1)

    # Calling ``stop()`` ensures that there is no residual movement
    move_group.stop()

    ## END_SUB_TUTORIAL

    # For testing:
    current_joints = move_group.get_current_joint_values()
    return all_close(joint_goal, current_joints, 0.01)

  def go_to_disengagePose(self):
    # Copy class variables to local variables to make the web tutorials more clear.
    # In practice, you should use the class variables directly unless you have a good
    # reason not to.
    move_group = self.move_group

    ## BEGIN_SUB_TUTORIAL plan_to_pose
    ##
    ## Planning to a Pose Goal
    ## ^^^^^^^^^^^^^^^^^^^^^^^
    ## We can plan a motion for this group to a desired pose for the
    ## end-effector:
    #     
    eef_pose_goal = eff_offsetCal.endEffectorPoseFromRobot(self.move_group.get_current_pose().pose)

    # Then change the orientation and other coordinates to the pose.
    setOrientation = tf.transformations.quaternion_from_euler(self.initOrientaionSXYZ[0],self.initOrientaionSXYZ[1],self.initOrientaionSXYZ[2],'sxyz') #static (s) rotating (r)

    eef_pose_goal.orientation.x = setOrientation[0]
    eef_pose_goal.orientation.y = setOrientation[1]
    eef_pose_goal.orientation.z = setOrientation[2]
    eef_pose_goal.orientation.w = setOrientation[3]
    # pose_goal.orientation.w = 1.0
    eef_pose_goal.position.x = self.engaging_endEffector_Position[0]
    eef_pose_goal.position.y = self.engaging_endEffector_Position[1]
    eef_pose_goal.position.z = self.engaging_endEffector_Position[2] + 0.10

    # move_group.set_pose_target(eff_offsetCal.robotPoseFromEndEffector(copy.deepcopy(eef_pose_goal)))



    #plan and execute gradual path change from current to target pose.
    robot_pose_goal = eff_offsetCal.robotPoseFromEndEffector(copy.deepcopy(eef_pose_goal))
    waypoints = self.getGradualWaypointsFromCurrent(robot_pose_goal)
    move_group.set_max_velocity_scaling_factor(0.15)
    move_group.set_max_acceleration_scaling_factor(0.15)
    (plan, fraction) = move_group.compute_cartesian_path(waypoints,0.005, 0)
    if len(plan.joint_trajectory.points)>30:
        print ("Retracting might look crazy")
    
    move_group.execute(plan, wait=True)    
    move_group.stop()    
    move_group.clear_pose_targets()

    ## END_SUB_TUTORIAL

    # For testing:
    # Note that since this section of code will not be included in the tutorials
    # we use the class variable rather than the copied state variable
    current_pose = self.move_group.get_current_pose().pose
    return all_close(robot_pose_goal, current_pose, 0.01)


  def get_engageEndEffectorPose(self):
    engagePose = geometry_msgs.msg.Pose()
    setOrientation = tf.transformations.quaternion_from_euler(self.initOrientaionSXYZ[0],self.initOrientaionSXYZ[1],self.initOrientaionSXYZ[2],'sxyz') #static (s) rotating (r)

    engagePose.orientation.x = setOrientation[0]
    engagePose.orientation.y = setOrientation[1]
    engagePose.orientation.z = setOrientation[2]
    engagePose.orientation.w = setOrientation[3]
    # engagePose.orientation.w = 1.0
    engagePose.position.x = self.engaging_endEffector_Position[0]
    engagePose.position.y = self.engaging_endEffector_Position[1]
    engagePose.position.z = self.engaging_endEffector_Position[2]

    return engagePose

  def go_to_engagePose(self, speedScale=0.5):
    # Copy class variables to local variables to make the web tutorials more clear.
    # In practice, you should use the class variables directly unless you have a good
    # reason not to.
    move_group = self.move_group
    robot = self.robot

    ## BEGIN_SUB_TUTORIAL plan_to_pose
    ##
    ## Planning to a Pose Goal
    ## ^^^^^^^^^^^^^^^^^^^^^^^
    ## We can plan a motion for this group to a desired pose for the
    ## end-effector:
    #     
    pose_goal =eff_offsetCal.robotPoseFromEndEffector(self.get_engageEndEffectorPose())

    waypoints = self.getGradualWaypointsFromCurrent(eff_offsetCal.robotPoseFromEndEffector(pose_goal))
    (plan, fraction) = move_group.compute_cartesian_path(waypoints,0.001, 0)
    if len(plan.joint_trajectory.points)>50:
        print ("Retracting might look crazy")

    # move_group.set_max_velocity_scaling_factor(speedScale)
    # move_group.set_max_acceleration_scaling_factor(speedScale)
    plan = move_group.retime_trajectory(robot.get_current_state(),plan,speedScale)  # This is what actually change the speed.
   
    move_group.execute(plan, wait=True)

    # print("Pose Goal-Engage")
    # print(pose_goal)
    # move_group.set_pose_target(pose_goal)


    # Set the speed to be lower than execution
    # move_group.set_max_velocity_scaling_factor(0.15)
    # move_group.set_max_acceleration_scaling_factor(0.15)

    

    ## Now, we call the planner to compute the plan and execute it.
    
    # move_group.plan(plan) # for debugging
    # plan = move_group.go(wait=True) # for debugging

    # # Restore the Max Velocity & Acceleration
    # move_group.set_max_velocity_scaling_factor(1)
    # move_group.set_max_acceleration_scaling_factor(1)
    # move_group.plan()


    # Calling `stop()` ensures that there is no residual movement
    move_group.stop()
    # It is always good to clear your targets after planning with poses.
    # Note: there is no equivalent function for clear_joint_value_targets()
    move_group.clear_pose_targets()

    
    

    ## END_SUB_TUTORIAL

    # For testing:
    # Note that since this section of code will not be included in the tutorials
    # we use the class variable rather than the copied state variable
    current_pose = self.move_group.get_current_pose().pose
    return all_close(pose_goal, current_pose, 0.01)

  def go_to_disengagePose_simple(self):
    return self.go_to_disengagePose()

    # # Copy class variables to local variables to make the web tutorials more clear.
    # # In practice, you should use the class variables directly unless you have a good
    # # reason not to.
    # move_group = self.move_group

    # ## BEGIN_SUB_TUTORIAL plan_to_pose
    # ##
    # ## Planning to a Pose Goal
    # ## ^^^^^^^^^^^^^^^^^^^^^^^
    # ## We can plan a motion for this group to a desired pose for the
    # ## end-effector:
    # #     
    # pose_goal = geometry_msgs.msg.Pose()

    # setOrientation = tf.transformations.quaternion_from_euler(self.initOrientaionSXYZ[0],self.initOrientaionSXYZ[1],self.initOrientaionSXYZ[2],'sxyz') #static (s) rotating (r)

    # pose_goal.orientation.x = setOrientation[0]
    # pose_goal.orientation.y = setOrientation[1]
    # pose_goal.orientation.z = setOrientation[2]
    # pose_goal.orientation.w = setOrientation[3]
    # # pose_goal.orientation.w = 1.0
    # pose_goal.position.x = self.engaging_endEffector_Position[0]
    # pose_goal.position.y = self.engaging_endEffector_Position[1]
    # pose_goal.position.z = self.engaging_endEffector_Position[2]+0.10

    # move_group.set_pose_target(eff_offsetCal.robotPoseFromEndEffector(pose_goal))


    # # Set the speed to be lower than execution
    # move_group.set_max_velocity_scaling_factor(0.3)
    # move_group.set_max_acceleration_scaling_factor(0.3)

    

    # ## Now, we call the planner to compute the plan and execute it.
    # plan = move_group.go(wait=True)

    # # # Restore the Max Velocity & Acceleration
    # # move_group.set_max_velocity_scaling_factor(1)
    # # move_group.set_max_acceleration_scaling_factor(1)
    # # move_group.plan()


    # # Calling `stop()` ensures that there is no residual movement
    # move_group.stop()
    # # It is always good to clear your targets after planning with poses.
    # # Note: there is no equivalent function for clear_joint_value_targets()
    # move_group.clear_pose_targets()

    
    

    # ## END_SUB_TUTORIAL

    # # For testing:
    # # Note that since this section of code will not be included in the tutorials
    # # we use the class variable rather than the copied state variable
    # current_pose = self.move_group.get_current_pose().pose
    # return all_close(pose_goal, current_pose, 0.01)

  




  def getRotatedRobotPose(self, robotPose, RotatingAngle, TranslateVector, rotateAxis=tf.transformations.random_vector(3)):
    waypoints = []
    waypoints.append(copy.deepcopy(robotPose))
    
    endEffPose = eff_offsetCal.endEffectorPoseFromRobot(robotPose)
    
    # If not defined, rotateAxis is randonly chosen
    # rotateAxis = tf.transformations.random_vector(3) 
    
    
    iterateNum = 5 # Incremental way poit for smoother path
    #1. Select a random axis of a quaternion, The angle will be fixed to 45deg    
    rotatingQuatern = tf.transformations.quaternion_about_axis(RotatingAngle/iterateNum, rotateAxis)    
    #2. multiply quaternion rotation to the current orientation.
    currQuatern = [endEffPose.orientation.x, endEffPose.orientation.y, endEffPose.orientation.z, endEffPose.orientation.w]    

    #3. iterate to have incremental waypoints.    
    for i in range(0,iterateNum):      
      newQuatern = tf.transformations.quaternion_multiply(rotatingQuatern, currQuatern)
      currQuatern = newQuatern

      #3. Update new EffPose
          
      endEffPose.orientation.x = newQuatern[0]
      endEffPose.orientation.y = newQuatern[1]
      endEffPose.orientation.z = newQuatern[2]
      endEffPose.orientation.w = newQuatern[3]
      
      
      endEffPose.position.x += (TranslateVector[0] / iterateNum)
      endEffPose.position.y += (TranslateVector[1] / iterateNum)
      endEffPose.position.z += (TranslateVector[2] / iterateNum)
            
      newRobotPose = eff_offsetCal.robotPoseFromEndEffector(copy.deepcopy(endEffPose))            
      waypoints.append(copy.deepcopy(newRobotPose))
      
    finalRobotPose = newRobotPose

    return waypoints, finalRobotPose  # new robot Pose

  def getGradualWaypointsFromCurrent(self, goalPose, iterateNum=5):
    move_group = self.move_group    
    currentPose = move_group.get_current_pose().pose
    
    waypoints = []
    waypoints.append(copy.deepcopy(currentPose))    
    v1 = numpy.zeros((3, ), dtype=numpy.float64)
    q1 = numpy.zeros((4, ), dtype=numpy.float64)
    v1[0] = currentPose.position.x
    v1[1] = currentPose.position.y
    v1[2] = currentPose.position.z
    q1[0] = currentPose.orientation.x
    q1[1] = currentPose.orientation.y
    q1[2] = currentPose.orientation.z
    q1[3] = currentPose.orientation.w

    v2 = numpy.zeros((3, ), dtype=numpy.float64)
    q2 = numpy.zeros((4, ), dtype=numpy.float64)
    v2[0] = goalPose.position.x
    v2[1] = goalPose.position.y
    v2[2] = goalPose.position.z
    q2[0] = goalPose.orientation.x
    q2[1] = goalPose.orientation.y
    q2[2] = goalPose.orientation.z
    q2[3] = goalPose.orientation.w

    tempPose = geometry_msgs.msg.Pose()

    for idx in range(0,iterateNum+1):
      fraction = idx / iterateNum
      q_temp = tf.transformations.quaternion_slerp(q1, q2, fraction)
      v_temp = v1 + (v2 - v1) * fraction
      
      tempPose.position.x = v_temp[0]
      tempPose.position.y = v_temp[1]
      tempPose.position.z = v_temp[2]
      tempPose.orientation.x = q_temp[0]
      tempPose.orientation.y = q_temp[1]
      tempPose.orientation.z = q_temp[2]
      tempPose.orientation.w = q_temp[3]

      waypoints.append(copy.deepcopy(tempPose))

    return waypoints # new robot Pose


  def plan_execute_cartesian_path(self, listIdx=-1):
    # Copy class variables to local variables to make the web tutorials more clear.
    # In practice, you should use the class variables directly unless you have a good
    # reason not to.
    move_group = self.move_group

    

    ## BEGIN_SUB_TUTORIAL plan_cartesian_path
    ##
    ## Cartesian Paths
    ## ^^^^^^^^^^^^^^^
    ## You can plan a Cartesian path directly by specifying a list of waypoints
    ## for the end-effector to go through. If executing  interactively in a
    ## Python shell, set scale = 1.0.
    ##
    

    robotPose = move_group.get_current_pose().pose
    
    if listIdx < 0: #No Index is given
      TranslateVector = [0.0, 0.0, 0.02]
      rotateAxis = [1,-1,0]
      RotatingAngle = pi/8;
    else:
      theta = self.motionList[listIdx][0]
      phi = self.motionList[listIdx][1]
            
      rotateAxis = [sin(phi)*cos(theta),sin(phi)*sin(theta),cos(phi)]
      TranslateVector = list(self.motionList[listIdx][2:])      
      RotatingAngle = pi/4;

    print("Translate Vector", end=' ')
    print(TranslateVector, end=' ')
    print("Rotating Axis", end=' ')
    print(rotateAxis)

    (waypoints, finalRobotPose) = self.getRotatedRobotPose(robotPose, RotatingAngle, TranslateVector, rotateAxis)

    # We want the Cartesian path to be interpolated at a resolution of 1 cm
    # which is why we will specify 0.01 as the eef_step in Cartesian
    # translation.  We will disable the jump threshold by setting it to 0.0,
    # ignoring the check for infeasible jumps in joint space, which is sufficient
    # for this tutorial.
    # Set the speed to be lower than execution
    
    (plan, fraction) = move_group.compute_cartesian_path(
                                       waypoints,   # waypoints to follow
                                       0.01,        # eef_step
                                       0.0)         # jump_threshold
    if len(plan.joint_trajectory.points) > 30:
      print("plan may too long: ", len(plan.joint_trajectory.points))
    move_group.execute(plan, wait=True)
    # rospy.sleep(1)

    move_group.stop()    
    move_group.clear_pose_targets()

    current_pose = self.move_group.get_current_pose().pose
    
    return all_close(finalRobotPose, current_pose, 0.01)
    # return plan, fraction

    ## END_SUB_TUTORIAL


  def display_trajectory(self, plan):
    # Copy class variables to local variables to make the web tutorials more clear.
    # In practice, you should use the class variables directly unless you have a good
    # reason not to.
    robot = self.robot
    display_trajectory_publisher = self.display_trajectory_publisher

    ## BEGIN_SUB_TUTORIAL display_trajectory
    ##
    ## Displaying a Trajectory
    ## ^^^^^^^^^^^^^^^^^^^^^^^
    ## You can ask RViz to visualize a plan (aka trajectory) for you. But the
    ## group.plan() method does this automatically so this is not that useful
    ## here (it just displays the same trajectory again):
    ##
    ## A `DisplayTrajectory`_ msg has two primary fields, trajectory_start and trajectory.
    ## We populate the trajectory_start with our current robot state to copy over
    ## any AttachedCollisionObjects and add our plan to the trajectory.
    display_trajectory = moveit_msgs.msg.DisplayTrajectory()
    display_trajectory.trajectory_start = robot.get_current_state()
    display_trajectory.trajectory.append(plan)
    # Publish
    display_trajectory_publisher.publish(display_trajectory);

    ## END_SUB_TUTORIAL


  def execute_plan(self, plan):
    # Copy class variables to local variables to make the web tutorials more clear.
    # In practice, you should use the class variables directly unless you have a good
    # reason not to.
    move_group = self.move_group

    ## BEGIN_SUB_TUTORIAL execute_plan
    ##
    ## Executing a Plan
    ## ^^^^^^^^^^^^^^^^
    ## Use execute if you would like the robot to follow
    ## the plan that has already been computed:
    move_group.execute(plan, wait=True)

    ## **Note:** The robot's current joint state must be within some tolerance of the
    ## first waypoint in the `RobotTrajectory`_ or ``execute()`` will fail
    ## END_SUB_TUTORIAL


  def wait_for_state_update(self, box_is_known=False, box_is_attached=False, timeout=2):
    # Copy class variables to local variables to make the web tutorials more clear.
    # In practice, you should use the class variables directly unless you have a good
    # reason not to.
    box_name = self.box_name
    scene = self.scene

    ## BEGIN_SUB_TUTORIAL wait_for_scene_update
    ##
    ## Ensuring Collision Updates Are Receieved
    ## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ## If the Python node dies before publishing a collision object update message, the message
    ## could get lost and the box will not appear. To ensure that the updates are
    ## made, we wait until we see the changes reflected in the
    ## ``get_attached_objects()`` and ``get_known_object_names()`` lists.
    ## For the purpose of this tutorial, we call this function after adding,
    ## removing, attaching or detaching an object in the planning scene. We then wait
    ## until the updates have been made or ``timeout`` seconds have passed
    start = rospy.get_time()
    seconds = rospy.get_time()
    while (seconds - start < timeout) and not rospy.is_shutdown():
      # Test if the box is in attached objects
      attached_objects = scene.get_attached_objects([box_name])
      is_attached = len(list(attached_objects.keys())) > 0

      # Test if the box is in the scene.
      # Note that attaching the box will remove it from known_objects
      is_known = box_name in scene.get_known_object_names()

      # Test if we are in the expected state
      if (box_is_attached == is_attached) and (box_is_known == is_known):
        return True

      # Sleep so that we give other threads time on the processor
      rospy.sleep(0.1)
      seconds = rospy.get_time()

    # If we exited the while loop without returning then we timed out
    return False
    ## END_SUB_TUTORIAL

  def add_box(self, name, position=(0,0,0), size=(0, 0, 0), timeout=2):  
    #Add the table under the robot in the scene so MoveIt is aware of it.
    #Table length = 60 in., Table depth = 30 in., Table height = 36.25 in.
    #Position vectors pretty much always describes (X,Y,Z) in this order.
    #The axis are defined by the base frame of the robot relative to which
    #everything is.

    #The position is the one of the center of the box (not a corner) and
    #is relative to the base frame of the robot. Also, the robot base is on
    #top of a 0.5 in. thick plate, is centered along the X axis but offset
    #by 5 in. along the Y axis.
    box_pose = geometry_msgs.msg.PoseStamped()
    box_pose.header.frame_id = self.robot.get_planning_frame()
    box_pose.pose.position.x = position[0]
    box_pose.pose.position.y = position[1]
    box_pose.pose.position.z = position[2]

    self.scene.add_box(name, box_pose, size=size)
    #DO NOT REMOVE this delay, its needed
    # to let the PlanningSceneInterface update.
    self.wait_for_state_update(box_is_known=False, timeout=timeout)

    print("Objects added to MoveIt planning scene:")
    print((self.scene.get_known_object_names()))

  def add_WallBondary(self, Dx, Dy, Dz, Lx, Ly, Lz, timeout=2):
    thickness = 0.01
    c1_size = (abs(Lx), thickness, abs(Lz))
    c2_size = (thickness, abs(Ly), abs(Lz))
    c3_size = (abs(Lx), thickness, abs(Lz))
    c4_size = (thickness, abs(Ly), abs(Lz))

    if(Lx == 0 or Ly == 0 or Lz == 0):
        print("ERROR: A length cannot be zero.")
        return

    sign_x = Lx/abs(Lx)
    sign_y = Ly/abs(Ly)
    sign_z = Lz/abs(Lz)

    c1_position = (Dx+Lx/2,                  Dy+sign_y*thickness/2,     Dz+Lz/2)
    c4_position = (Dx+sign_x*thickness/2,    Dy+Ly/2,                   Dz+Lz/2)
    c3_position = (Dx+Lx/2,                  Dy+Ly-sign_y*thickness/2,  Dz+Lz/2)
    c2_position = (Dx+Lx-sign_x*thickness/2, Dy+Ly/2,                   Dz+Lz/2)

    WallName = "wall"
    self.add_box(WallName+"_c1",c1_position, c1_size)
    self.add_box(WallName+"_c2",c2_position, c2_size)
    self.add_box(WallName+"_c3",c3_position, c3_size)   

    self.wait_for_state_update(box_is_known=False, timeout=timeout) 


  def add_sphere(self, timeout=2):
    # Copy class variables to local variables to make the web tutorials more clear.
    # In practice, you should use the class variables directly unless you have a good
    # reason not to.
    box_name = self.box_name
    scene = self.scene
    
    if True: #not(scene.get_attached_objects()):
      ## BEGIN_SUB_TUTORIAL add_box
      ##
      ## Adding Objects to the Planning Scene
      ## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      ## First, we will create a box in the planning scene at the location of the left finger:
      box_pose = geometry_msgs.msg.PoseStamped()
      box_pose.header.frame_id = "ee_link"
      box_pose.pose.orientation.w = 1.0
      box_pose.pose.position.x = self.endEffector_offset[0] # slightly above the end effector
      box_pose.pose.position.y = self.endEffector_offset[1]
      box_pose.pose.position.z = self.endEffector_offset[2]

      box_name = "box"
      scene.add_sphere(box_name, box_pose, radius=0.02)
      # scene.add_box(box_name, box_pose, size=(0.05, 0.05, 0.05))

      ## END_SUB_TUTORIAL
      # Copy local variables back to class variables. In practice, you should use the class
      # variables directly unless you have a good reason not to.
      self.box_name=box_name
      return self.wait_for_state_update(box_is_known=False, timeout=timeout)


  def attach_sphere(self, timeout=2):
    # Copy class variables to local variables to make the web tutorials more clear.
    # In practice, you should use the class variables directly unless you have a good
    # reason not to.
    box_name = self.box_name
    robot = self.robot
    scene = self.scene
    if True: #not(scene.get_attached_objects()):
      eef_link = self.eef_link
      group_names = self.group_names

      ## BEGIN_SUB_TUTORIAL attach_object
      ##
      ## Attaching Objects to the Robot
      ## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      ## Next, we will attach the box to the Panda wrist. Manipulating objects requires the
      ## robot be able to touch them without the planning scene reporting the contact as a
      ## collision. By adding link names to the ``touch_links`` array, we are telling the
      ## planning scene to ignore collisions between those links and the box. For the Panda
      ## robot, we set ``grasping_group = 'hand'``. If you are using a different robot,
      ## you should change this value to the name of your end effector group name.
      grasping_group_name = 'endeffector'
      touch_links = robot.get_link_names(group=grasping_group_name)
      scene.attach_box(eef_link, box_name, touch_links=touch_links)
      ## END_SUB_TUTORIAL   


      # We wait for the planning scene to update.
      return self.wait_for_state_update(box_is_attached=True, box_is_known=False, timeout=timeout)


  def detach_box(self, timeout=2):
    # Copy class variables to local variables to make the web tutorials more clear.
    # In practice, you should use the class variables directly unless you have a good
    # reason not to.
    box_name = self.box_name
    scene = self.scene
    eef_link = self.eef_link

    ## BEGIN_SUB_TUTORIAL detach_object
    ##
    ## Detaching Objects from the Robot
    ## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ## We can also detach and remove the object from the planning scene:
    scene.remove_attached_object(eef_link, name=box_name)
    ## END_SUB_TUTORIAL

    # We wait for the planning scene to update.
    return self.wait_for_state_update(box_is_known=False, box_is_attached=False, timeout=timeout)


  def remove_box(self, timeout=2):
    # Copy class variables to local variables to make the web tutorials more clear.
    # In practice, you should use the class variables directly unless you have a good
    # reason not to.
    box_name = self.box_name
    scene = self.scene

    ## BEGIN_SUB_TUTORIAL remove_object
    ##
    ## Removing Objects from the Planning Scene
    ## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ## We can remove the box from the world.
    scene.remove_world_object(box_name)
    

    ## **Note:** The object must be detached before we can remove it from the world
    ## END_SUB_TUTORIAL

    # We wait for the planning scene to update.
    return self.wait_for_state_update(box_is_attached=False, box_is_known=False, timeout=timeout)


  def go_to_Position(self,targetRobotPosition,speedScale=1.0,wantWait = False):
    #translate the endeffector with fixed orientation

    move_group = self.move_group
    robot = self.robot

    ## BEGIN_SUB_TUTORIAL plan_to_pose
    ##
    ## Planning to a Pose Goal
    ## ^^^^^^^^^^^^^^^^^^^^^^^
    ## We can plan a motion for this group to a desired pose for the
    ## end-effector:
    #     
    pose_goal = geometry_msgs.msg.Pose()

    setOrientation = tf.transformations.quaternion_from_euler(self.initOrientaionSXYZ[0],self.initOrientaionSXYZ[1],self.initOrientaionSXYZ[2],'sxyz') #static (s) rotating (r)

    pose_goal.orientation.x = setOrientation[0]
    pose_goal.orientation.y = setOrientation[1]
    pose_goal.orientation.z = setOrientation[2]
    pose_goal.orientation.w = setOrientation[3]
    # pose_goal.orientation.w = 1.0
    pose_goal.position.x = targetRobotPosition[0]
    pose_goal.position.y = targetRobotPosition[1]
    pose_goal.position.z = targetRobotPosition[2]

    # move_group.set_pose_target(eff_offsetCal.robotPoseFromEndEffector(pose_goal))

    
    
    waypoints = self.getGradualWaypointsFromCurrent(eff_offsetCal.robotPoseFromEndEffector(pose_goal))
    (plan, fraction) = move_group.compute_cartesian_path(waypoints,0.001, 0)
    if len(plan.joint_trajectory.points)>50:
        print ("Retracting might look crazy")

    # move_group.set_max_velocity_scaling_factor(speedScale)
    # move_group.set_max_acceleration_scaling_factor(speedScale)
    plan = move_group.retime_trajectory(robot.get_current_state(),plan,speedScale)  # This is what actually change the speed.
    
    
    move_group.execute(plan, wait=wantWait)    


    # # Set the speed to be lower than execution
    # move_group.set_max_velocity_scaling_factor(speedScale)
    # move_group.set_max_acceleration_scaling_factor(speedScale)

    # plan = move_group.plan()
    # print len(plan.joint_trajectory.points)
    # if len(plan.joint_trajectory.points)>5:
    #     print ("Retracting might look crazy")
    #     # return 0

    # move_group.go(wait=wantWait)

    if wantWait:
      move_group.stop()
      move_group.clear_pose_targets()  
      
      current_pose = move_group.get_current_pose().pose
      return all_close(pose_goal, current_pose, 0.01)
    else:
      return 0

    
  def stop_and_clear(self):
    ## Now, we call the planner to compute the plan and execute it.
    move_group = self.move_group
    # Calling `stop()` ensures that there is no residual movement
    move_group.stop()
    # It is always good to clear your targets after planning with poses.
    # Note: there is no equivalent function for clear_joint_value_targets()
    move_group.clear_pose_targets()  

  def go_to_goal_jointState(self, targetJointState, speed = 0.3):
    # Copy class variables to local variables to make the web tutorials more clear.
    # In practice, you should use the class variables directly unless you have a good
    # reason not to.
    move_group = self.move_group

    ## BEGIN_SUB_TUTORIAL plan_to_joint_state
    ##
    ## Planning to a Joint Goal
    ## ^^^^^^^^^^^^^^^^^^^^^^^^
    ## The Panda's zero configuration is at a `singularity <https://www.quora.com/Robotics-What-is-meant-by-kinematic-singularity>`_ so the first
    ## thing we want to do is move it to a slightly better configuration.
    # We can get the joint values from the group and adjust some of the values:
    joint_goal = targetJointState
    
    move_group.set_max_velocity_scaling_factor(speed)
    move_group.set_max_acceleration_scaling_factor(speed)
    
    # The go command can be called with joint values, poses, or without any
    # parameters if you have already set the pose or joint target for the group
    move_group.go(joint_goal, wait=True)
    # move_group.go(joint_goal, wait=True)

    rospy.sleep(0.1)

    # Calling ``stop()`` ensures that there is no residual movement
    move_group.stop()

    ## END_SUB_TUTORIAL

    # For testing:
    current_joints = move_group.get_current_joint_values()
    return all_close(joint_goal, current_joints, 0.01)
    

  def go_to_PoseGradually(self,robotPoseGoal,speedScale=1.0,wantWait = False,segmentNum = 10):
    #translate the endeffector with fixed orientation

    move_group = self.move_group
    robot = self.robot
    ## BEGIN_SUB_TUTORIAL plan_to_pose
    ##
    ## Planning to a Pose Goal
    ## ^^^^^^^^^^^^^^^^^^^^^^^
    ## We can plan a motion for this group to a desired pose for the
    ## end-effector:
    #     
    
    # robotPoseGoal = geometry_msgs.msg.Pose()
    # move_group.set_pose_target(eff_offsetCal.robotPoseFromEndEffector(robotPoseGoal))


    waypoints = self.getGradualWaypointsFromCurrent(robotPoseGoal,iterateNum=segmentNum)
    (plan, fraction) = move_group.compute_cartesian_path(waypoints,0.001, 0)
    print(len(plan.joint_trajectory.points))
    if len(plan.joint_trajectory.points)>50:
        print ("Retracting might look crazy")

    # move_group.set_max_velocity_scaling_factor(speedScale)
    # move_group.set_max_acceleration_scaling_factor(speedScale)
        
    plan = move_group.retime_trajectory(robot.get_current_state(),plan,speedScale)  # This is what actually change the speed.
    
    move_group.execute(plan, wait=wantWait)    

    if wantWait:
      move_group.stop()
      move_group.clear_pose_targets()  
      
      current_pose = move_group.get_current_pose().pose
      return all_close(robotPoseGoal, current_pose, 0.001)
    else:
      return 0
   