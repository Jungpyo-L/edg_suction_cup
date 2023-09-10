#!/usr/bin/env python
import os
import pickle
from scipy.spatial.transform import Rotation as R
import numpy as np

from visualization import Visualizer3D as vis3d
from autolab_core import Logger, Point, DepthImage
from sensor_msgs.msg import PointCloud2
from ros_numpy import point_cloud2 as pc2
from PIL import Image, ImageDraw

import perception

import sys
sys.path.append('/home/edg/catkin_ws_new/src/Kinect_Smoothing')
from kinect_smoothing import HoleFilling_Filter, Denoising_Filter


#Camera Intrinsics
datadir = '/home/edg/TaeExperiment/tmpPlannedGrasp/220718/131501'
with open(datadir + '/gqcnnResult.p', 'rb') as handle:
   loaded_data = pickle.load(handle)
camera_intr = loaded_data['camera_intr']

# real data
# datadir = '/home/edg/TaeExperiment/tmpPlannedGrasp/220715/161545'
# datadir = '/home/edg/TaeExperiment/tmpPlannedGrasp/220715/161740'
# datadir = '/home/edg/TaeExperiment/tmpPlannedGrasp/220715/161816'
# datadir = '/home/edg/TaeExperiment/tmpPlannedGrasp/220715/161902'  # Flat box



# datadir = '/home/edg/TaeExperiment/tmpPlannedGrasp/220718/131501'  # box
datadir = '/home/edg/TaeExperiment/tmpPlannedGrasp/220718/131613' # 3D Printed
# datadir = '/home/edg/TaeExperiment/tmpPlannedGrasp/220718/131657'   # Cylinder
with open(datadir + '/gqcnnResult.p', 'rb') as handle:
   loaded_data = pickle.load(handle)
depth_im = loaded_data['depth_im']


# inpaint just look the same.
depth_im = depth_im.inpaint()

testFilter = False
testNormalCloud = True

if testFilter:# Test filter
   noise_filter = Denoising_Filter(flag='gaussian', ksize=9, sigma=0.5) 
   # noise_filter = Denoising_Filter(flag='anisotropic', theta=60) 

   denoise_image_frame = noise_filter.smooth_image_frames([depth_im.data])
   depth_im_filtered = DepthImage(denoise_image_frame[0],frame=depth_im.frame)


   pointCloud_raw = camera_intr.deproject(depth_im).data.T
   validIdx = np.logical_and(pointCloud_raw[:,2] < 0.32, pointCloud_raw[:,2] > 0.029)

   # pointCloud = camera_intr.deproject(depth_im).data.T
   pointCloud = camera_intr.deproject(depth_im_filtered).data.T
   validPointCloud = pointCloud[validIdx,:]


   vis3d.figure()
   vis3d.points(validPointCloud, scale=0.0005)

   for pose in loaded_data['poses']:
      addedPoint = [pose.position.x, pose.position.y, pose.position.z]
      r = R.from_quat([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
      rotMat = r.as_matrix()
      vect = rotMat[:,0]

      vis3d.arrow(addedPoint, -vect/50, tube_radius=1e-3, color = (1.0, 0, 0))

   # vis3d.pose() # to show the reference axis
   vis3d.show()

if testNormalCloud:# Test normal Cloudfilter

   pointNormalCloud = depth_im.point_normal_cloud(camera_intr)  # great if we to train for random normal vects.    
   pointCloud_raw = pointNormalCloud.point_cloud.data.T


   # img = Image.new('L', (width, height), 0)
   # ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
   # mask = numpy.array(img)


   # Filter Image
   noise_filter = Denoising_Filter(flag='gaussian', ksize=9, sigma=0.5) 
   # noise_filter = Denoising_Filter(flag='anisotropic', theta=60) 
   denoise_image_frame = noise_filter.smooth_image_frames([depth_im.data])
   depth_im = DepthImage(denoise_image_frame[0],frame=depth_im.frame)



   pointNormalCloud = depth_im.point_normal_cloud(camera_intr)  # great if we to train for random normal vects.    
   pointCloud = pointNormalCloud.point_cloud.data.T

   distVect = np.linalg.norm(pointCloud-pointCloud_raw,axis=1)
   dispCriteria = distVect < 1e-3

   validIdx = np.logical_and(pointCloud[:,2] < 0.32, pointCloud[:,2] > 0.25)
   validIdx = np.logical_and(validIdx, dispCriteria) # to filterout the edges

   # pointCloud = camera_intr.deproject(depth_im).data.T
   # pointCloud = camera_intr.deproject(depth_im_filtered).data.T
   validPointCloud_raw = pointCloud_raw[validIdx,:]
   validPointCloud = pointCloud[validIdx,:]
   validNormalVects = pointNormalCloud.normal_cloud.data.T[validIdx,:]

   vis3d.figure()
   
   randomI = np.random.permutation(len(validPointCloud))

   edgeRegionCrit = 2e-3
   numberThres = 45

   for idx in randomI[0:100]:   
      addedPoint = validPointCloud[idx,:]
      numPointsInRegion = np.sum(np.linalg.norm(validPointCloud-addedPoint, axis=1) < edgeRegionCrit)
      print(numPointsInRegion)
      if numPointsInRegion > numberThres:
         vect = validNormalVects[idx,:]
         vis3d.arrow(addedPoint, vect/60, tube_radius=0.5e-3, color = (1.0, 0, 0))

   
   vis3d.points(validPointCloud, scale=0.0005)
   vis3d.show()


raise('Stop')

camera_intr.deproject_pixel(depth_im[0,0], Point(np.array([0,0]), frame=camera_intr.frame))


pc2.array_to_pointcloud2(loaded_data['depth_im'].data)







# Load saved Trasnform matrix        
datadir = os.path.dirname(os.path.realpath(__file__))
with open(datadir + '/Pose_sample_binder', 'rb') as handle:
   poses = pickle.load(handle)

thisPose = poses[0]
thisQuat = [thisPose.orientation.x, thisPose.orientation.y, thisPose.orientation.z, thisPose.orientation.w]

r_pose_from_cam = R.from_quat(thisQuat)
axis_in_cam = r_pose_from_cam.as_matrix()
print(axis_in_cam)
targetVec = axis_in_cam[:,0] # rotated x is the target vector


# Load saved Trasnform matrix        
datadir = os.path.dirname(os.path.realpath(__file__))
with open(datadir + '/TransformMat_board_verified', 'rb') as handle:
   loaded_data = pickle.load(handle)

T = loaded_data
R_N_cam = T[0:3,0:3]
targetVec_N = np.matmul(R_N_cam,targetVec)
print(targetVec_N)

# to us, z axis of the the tool0 should be the 
# currZ_vec = 
