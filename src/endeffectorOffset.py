import numpy
import tf

# endEffector_offset = numpy.array([0.160, 0.0, 0.0]) # for incremental press Test

# endEffector_offset = numpy.array([0.140, 0.0, 0.0]) # x, y, z offset
# endEffector_offset = numpy.array([0.150, 0.0, 0.0]) # x, y, z offset # From Cad

# endEffector_offset = numpy.array([0.144, 0.0, 0.0]) # x, y, z offset # From Cad
endEffector_offset = numpy.array([0.0, 0.0, 0.0]) # x, y, z offset # burrow robot case fixture




def robotPoseFromEndEffector(desired_EFPose):    
    global endEffector_offset
    targetPose = desired_EFPose

    currOrientation = [desired_EFPose.orientation.x, desired_EFPose.orientation.y, desired_EFPose.orientation.z, desired_EFPose.orientation.w]
    R = tf.transformations.quaternion_matrix(currOrientation)    
    translation = numpy.dot(R[0:3,0:3], endEffector_offset)
    targetPose.position.x -= translation[0]
    targetPose.position.y -= translation[1]
    targetPose.position.z -= translation[2]

    return targetPose


def endEffectorPoseFromRobot(RobotPose):    
    global endEffector_offset
    targetPose = RobotPose
    
    currOrientation = [RobotPose.orientation.x, RobotPose.orientation.y, RobotPose.orientation.z, RobotPose.orientation.w]
    R = tf.transformations.quaternion_matrix(currOrientation)    
    translation = numpy.dot(R[0:3,0:3], endEffector_offset)
    targetPose.position.x += translation[0]
    targetPose.position.y += translation[1]
    targetPose.position.z += translation[2]

    return targetPose