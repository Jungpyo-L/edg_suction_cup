#!/usr/bin/env python
from autolab_core import Logger, Point, CameraIntrinsics, DepthImage, BinaryImage
import pickle
import numpy as np
import matplotlib.pyplot as plt

datadir = '/home/tae/TaeExperiment/tmpPlannedGrasp/220928/101405'

with open(datadir + '/gqcnnResult.p', 'rb') as handle:
       gqCNNData = pickle.load(handle)

with open(datadir + '/point_normalResult.p', 'rb') as handle:
       pointNormlData = pickle.load(handle)

depth_im = gqCNNData['depth_im']

BinMask = depth_im.data < -10
TopLeft = [10, 10] # x, y 
BottomRight = [400, 300] # x, y
for i in range(TopLeft[1], BottomRight[1]):
       for j in range(TopLeft[0],BottomRight[0]):
              BinMask[i,j] = True

segmask = BinaryImage(np.iinfo(np.uint8).max *
              (1 * ( (depth_im.data < 0.5) & BinMask)).astype(np.uint8) ,
              frame=depth_im.frame)















from scipy.io import loadmat
import scipy
import os
import numpy as np
from helperFunction.utils import rotation_from_quaternion, create_transform_matrix, quaternion_from_matrix, normalize, hat

phi = 0
thetaMax = 30
theta = np.pi/6

omega_hat = hat(np.array([np.cos(phi), np.sin(phi), 0]))
Rw = scipy.linalg.expm(theta * omega_hat)
T_from_tipContact = create_transform_matrix(Rw, [0.0, 0.0, 0.0]) 

folderName = os.path.expanduser('~') + '/TaeExperiment/221005_tabulated_p_psd_FT_image.mat'
# fileName = "221005_tabulated_p_psd_FT_image.mat"

mat_contents = loadmat(folderName)
print('here')






from mimetypes import init
from helperFunction.adaptiveMotion import adaptMotionHelp
from geometry_msgs.msg import PoseStamped, Pose
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import scipy
from src.helperFunction.utils import hat, create_transform_matrix
import rospy

from helperFunction.SuctionP_callback_helper import P_CallbackHelp

from helperFunction.rigid_transform_3D.rigid_transform_3D import rigid_transform_3D


a = np.empty((3,5))
# a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])
np.concatenate((a, b), axis=0)
array([[1, 2],
       [3, 4],
       [5, 6]])
np.concatenate((a, b.T), axis=1)
array([[1, 2, 5],
       [3, 4, 6]])
np.concatenate((a, b), axis=None)
array([1, 2, 3, 4, 5, 6])



# Random rotation and translation
R = np.random.rand(3,3)
t = np.random.rand(3,1)

# make R a proper rotation matrix, force orthonormal
U, S, Vt = np.linalg.svd(R)
R = U@Vt

# remove reflection
if np.linalg.det(R) < 0:
   Vt[2,:] *= -1
   R = U@Vt

# number of points
n = 10

A = np.random.rand(3, n)*0.2
B = R@A + t


mu, sigma = 0, 0.001 # mean and standard deviation
randNoise = np.random.normal(mu, sigma, (3,n))
B = B + randNoise

# Recover R and t
ret_R, ret_t = rigid_transform_3D(A, B)


# controller node
rospy.init_node('tae_test')

P_help = P_CallbackHelp() # it deals with subscription.
rospy.sleep(0.5)
P_help.startSampling()    
rospy.sleep(0.5)
P_help.setNowAsOffset()

P_vac = P_help.P_vac

P_help.stopSampling()








# def getGoalPoseFromGQCNN(T, thisPose, initEndEffPose ):
#   #From pose Information of GQCNN, get the transformed orientation of the UR10

#   initEndEffectorPosition, initEndEffectorQuat = initEndEffPose
#   deltaPosition = np.matmul( T, (np.array([thisPose.position.x, thisPose.position.y, thisPose.position.z, 1])))  
#   goalRobotPosition = deltaPosition[0:3] + np.array(initEndEffectorPosition)
  

#   # orient
#   thisQuat = [thisPose.orientation.x, thisPose.orientation.y, thisPose.orientation.z, thisPose.orientation.w]

#   r_pose_from_cam = Rot.from_quat(thisQuat)
#   axis_in_cam = r_pose_from_cam.as_matrix()
#   # print(axis_in_cam)
#   targetVec = axis_in_cam[:,0] # rotated x is the target vector

  
#   R_N_cam = T[0:3,0:3]
#   targetSuctionAxisVec_N = np.matmul(R_N_cam,targetVec)
#   # print(targetSuctionAxisVec_N)

#   # to us, z axis of the the tool0 should be the 

#   r_currOrient_RobotEff = Rot.from_quat(initEndEffectorQuat)
#   currSuctionAxisVec_N = r_currOrient_RobotEff.as_matrix()[:,2]

#   rotAxis = np.cross(currSuctionAxisVec_N, targetSuctionAxisVec_N)  
#   rotAxis /= np.linalg.norm(rotAxis)
#   angleBtwTwo = np.arccos(np.dot(currSuctionAxisVec_N, targetSuctionAxisVec_N))

#   r_RotOrient = Rot.from_mrp(rotAxis * np.tan(angleBtwTwo / 4))
#   r_targetOrient_RobotEff = r_RotOrient*r_currOrient_RobotEff
  
#   targetOrient_quat = r_targetOrient_RobotEff.as_quat()

#   return goalRobotPosition, targetOrient_quat, targetSuctionAxisVec_N

# def getGoalPoseFromGQCNN_re(T, thisPose, initEndEffPose ):
#   #From pose Information of GQCNN, get the transformed orientation of the UR10

#   initEndEffectorPosition, initEndEffectorQuat = initEndEffPose
#   deltaPosition = np.matmul( T, (np.array([thisPose.position.x, thisPose.position.y, thisPose.position.z, 1])))  
#   goalRobotPosition = deltaPosition[0:3] + np.array(initEndEffectorPosition)
  

#   # orient
#   thisQuat = [thisPose.orientation.x, thisPose.orientation.y, thisPose.orientation.z, thisPose.orientation.w]

#   r_pose_from_cam = Rot.from_quat(thisQuat)
#   axis_in_cam = r_pose_from_cam.as_matrix()
#   # print(axis_in_cam)
#   targetVec = axis_in_cam[:,0] # rotated x is the target vector

  
#   R_N_cam = T[0:3,0:3]
#   targetSuctionAxisVec_N = np.matmul(R_N_cam,targetVec)
#   # print(targetSuctionAxisVec_N)

#   # to us, z axis of the the tool0 should be the 

#   r_currOrient_RobotEff = Rot.from_quat(initEndEffectorQuat)
#   currSuctionAxisVec_N = r_currOrient_RobotEff.as_matrix()[:,2]

#   rotAxis = np.cross(currSuctionAxisVec_N, targetSuctionAxisVec_N)  
#   rotAxis /= np.linalg.norm(rotAxis)
#   angleBtwTwo = np.arccos(np.dot(currSuctionAxisVec_N, targetSuctionAxisVec_N))

#   rotAxis_in_BodyF = r_currOrient_RobotEff.apply(rotAxis, inverse=True)
  
#   r_RotOrient = Rot.from_mrp(rotAxis_in_BodyF * np.tan(angleBtwTwo / 4))
  

#   Rw = scipy.linalg.expm(angleBtwTwo * hat(rotAxis_in_BodyF))
#   print('=================')
#   print(r_RotOrient.as_matrix())
#   print(Rw)
#   print('=================')

  
#   r_targetOrient_RobotEff = r_currOrient_RobotEff*r_RotOrient
  
#   targetOrient_quat = r_targetOrient_RobotEff.as_quat()

#   return goalRobotPosition, targetOrient_quat, targetSuctionAxisVec_N  


# r = Rot.from_euler('xyz', [180, 0, 180], degrees=True)
# quat= r.as_quat()

# r_targ = Rot.from_euler('xyz',[4, 39, 32], degrees=True)
# quat_targ = r_targ.as_quat()
# thisPose = Pose()
# thisPose.orientation.x,thisPose.orientation.y, thisPose.orientation.z, thisPose.orientation.w = quat_targ 

# initEndEffPose = [[0,0,0], quat]
# T = np.eye(4)

# goalRobotPosition, targetOrient_quat, targetSuctionAxisVec_N = getGoalPoseFromGQCNN(T, thisPose, initEndEffPose)
# r3 = Rot.from_quat(targetOrient_quat)
# print(r3.as_matrix())


# goalRobotPosition, targetOrient_quat, targetSuctionAxisVec_N = getGoalPoseFromGQCNN_re(T, thisPose, initEndEffPose)
# r4 = Rot.from_quat(targetOrient_quat)
# print(r4.as_matrix())


# test = adaptMotionHelp()
# pose = PoseStamped()
# pose.pose.orientation.w = 1
# print(test.get_Tmat_from_Pose(pose))

# print(create_transform_matrix(np.eye(3), [0, 0, 15] ))




# # import os
# # import pickle
# # from scipy.spatial.transform import Rotation as R
# # import numpy as np

# # from visualization import Visualizer3D as vis3d
# # from autolab_core import Logger, Point
# # from sensor_msgs.msg import PointCloud2
# # from ros_numpy import point_cloud2 as pc2


# # #Camera Intrinsics
# # datadir = '/home/edg/TaeExperiment/tmpPlannedGrasp/220718/131501'
# # with open(datadir + '/gqcnnResult.p', 'rb') as handle:
# #    loaded_data = pickle.load(handle)
# # camera_intr = loaded_data['camera_intr']

# # # real data
# # datadir = '/home/edg/TaeExperiment/tmpPlannedGrasp/220715/161545'

# # # datadir = '/home/edg/TaeExperiment/tmpPlannedGrasp/220718/131501'  # box
# # # datadir = '/home/edg/TaeExperiment/tmpPlannedGrasp/220718/131613' # 3D Printed
# # # datadir = '/home/edg/TaeExperiment/tmpPlannedGrasp/220718/131657'   # Cylinder
# # with open(datadir + '/gqcnnResult.p', 'rb') as handle:
# #    loaded_data = pickle.load(handle)
# # depth_im = loaded_data['depth_im']



# # pointCloud = camera_intr.deproject(depth_im).data.T

# # validIdx = np.logical_and(pointCloud[:,2] < 0.31, pointCloud[:,2] > 0.03)

# # validPointCloud = pointCloud[validIdx,:]


# # vis3d.figure()
# # vis3d.points(validPointCloud, scale=0.0005)

# # for pose in loaded_data['poses']:
# #    addedPoint = [pose.position.x, pose.position.y, pose.position.z]
# #    r = R.from_quat([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
# #    rotMat = r.as_matrix()
# #    vect = rotMat[:,0]

# #    vis3d.arrow(addedPoint, -vect/50, tube_radius=1e-3)

# # # vis3d.pose() # to show the reference axis
# # vis3d.show()

# # raise('Stop')

# # camera_intr.deproject_pixel(depth_im[0,0], Point(np.array([0,0]), frame=camera_intr.frame))


# # pc2.array_to_pointcloud2(loaded_data['depth_im'].data)







# # # Load saved Trasnform matrix        
# # datadir = os.path.dirname(os.path.realpath(__file__))
# # with open(datadir + '/Pose_sample_binder', 'rb') as handle:
# #    poses = pickle.load(handle)

# # thisPose = poses[0]
# # thisQuat = [thisPose.orientation.x, thisPose.orientation.y, thisPose.orientation.z, thisPose.orientation.w]

# # r_pose_from_cam = R.from_quat(thisQuat)
# # axis_in_cam = r_pose_from_cam.as_matrix()
# # print(axis_in_cam)
# # targetVec = axis_in_cam[:,0] # rotated x is the target vector


# # # Load saved Trasnform matrix        
# # datadir = os.path.dirname(os.path.realpath(__file__))
# # with open(datadir + '/TransformMat_board_verified', 'rb') as handle:
# #    loaded_data = pickle.load(handle)

# # T = loaded_data
# # R_N_cam = T[0:3,0:3]
# # targetVec_N = np.matmul(R_N_cam,targetVec)
# # print(targetVec_N)

# # # to us, z axis of the the tool0 should be the 
# # # currZ_vec = 
