import os 
from pathlib import Path
from scipy.io import loadmat, savemat
import pickle
from autolab_core import PointCloudImage, DepthImage, BinaryImage, GrayscaleImage
from adaptiveMotion import adaptMotionHelp
import tf
import numpy as np
from geometry_msgs.msg import PoseStamped
from utils import rotation_from_quaternion as RFromq
import copy
from scipy.spatial.transform import Rotation as Rot

def point_cloud_to_image(points, imgsize, radius=1.0, num_sample=1, depthRange = 1e-2, minVal = 10):
    # Copyright (c) Facebook, Inc. and its affiliates.
    # 
    # This source code is licensed under the MIT license found in the
    # LICENSE file in the root directory of this source tree.
    # Tae Modified this code a bit.

    """ Utility functions for processing point clouds.
    Author: Charles R. Qi and Or Litany
    """
    """ input is Nx3 points
        output is imgsize*imgsize*num_sample*3
        assumes points are in range [-radius, radius]
        samples num_sample points in each pixel, if there are less than
        num_sample points, replicate the points
        Added on Feb 19
    """
    
    maxVal = 255-minVal
    img = np.zeros((imgsize, imgsize, 1),np.uint8)
    pixel = 2*radius/float(imgsize)
    locations = (points[:,0:2] + radius)/pixel # Nx2
    locations = locations.astype(int)
    loc2pc = {}
    for n in range(points.shape[0]):
        loc = tuple(locations[n,:])
        if loc not in loc2pc:
            loc2pc[loc] = []
        loc2pc[loc].append(points[n,:])
    for i in range(imgsize):
        for j in range(imgsize):
            if (i,j) not in loc2pc:
                val = 0 # grab only the depth component.
                # img[i,j,:,:] = np.zeros((num_sample,3))
            else:
                pc = loc2pc[(i,j)]
                pc = np.vstack(pc)
                if pc.shape[0]>=num_sample:
                    pc = np.min(pc,axis=0)
                    zDepth = pc[2]
                    if zDepth < -depthRange:
                        val = 255
                    elif zDepth > depthRange: #z increases with deep points 
                        val = 0
                    else:
                        val = (zDepth + depthRange)/(2*depthRange) * (minVal-maxVal) + maxVal                        
                    # pc = random_sampling(pc, num_sample, False)
                    
                    

                elif pc.shape[0]<num_sample:                    
                    val = 0 # grab only the depth component.
                    # pc = np.lib.pad(pc, ((0,num_sample-pc.shape[0]),(0,0)), 'edge')
                # pc_center = (np.array([i,j])+0.5)*pixel - radius
                # pc[:,0:2] = (pc[:,0:2] - pc_center)/pixel
                # img[i,j,:] = pc

                img[i,j,0] = np.uint8(val)
    
    return img

def random_sampling(pc, num_sample, replace=None, return_choices=False):
    """ Input is NxC, output is num_samplexC
    """
    if replace is None: replace = (pc.shape[0]<num_sample)
    choices = np.random.choice(pc.shape[0], num_sample, replace=replace)
    if return_choices:
        return pc[choices], choices
    else:
        return pc[choices]


#=================================================================
# 

def checkActualPoseIdx(thisPoseIdx, matData,poseList,adpt_help, T_cam, prevCorrection):
    # Compare it with the endeffector measure            
    p_xyz_measured = matData['endEffectorPose_data'][1,5:]
    q_wxyz = matData['endEffectorPose_data'][1,1:5]
    # q_wxyz = np.mean(matData['endEffectorPose_data'][:,1:5],axis = 0)
    q_xyzw_measured = q_wxyz[[1,2,3,0]]
    r_measured = Rot.from_quat(q_xyzw_measured)

    disengagePosition =  [-0.59, -.105, 0.26] # When depth is 0 cm. unit is in m
    setOrientation = tf.transformations.quaternion_from_euler(np.pi,0,np.pi/2,'sxyz') #static (s) rotating (r)
    initEndEffPoseStamped_actual = PoseStamped()                  
    initEndEffPoseStamped_actual.pose.orientation.x = setOrientation[0]
    initEndEffPoseStamped_actual.pose.orientation.y = setOrientation[1]
    initEndEffPoseStamped_actual.pose.orientation.z = setOrientation[2]
    initEndEffPoseStamped_actual.pose.orientation.w = setOrientation[3]    
    initEndEffPoseStamped_actual.pose.position.x = -0.59
    initEndEffPoseStamped_actual.pose.position.y = -.105
    initEndEffPoseStamped_actual.pose.position.z = 0.26

    positionDiff_list = []
    AngleDiff_list = []
    for testPose in poseList:
        targetPoseStamped_actual = adpt_help.getGoalPosestampedFromGQCNN(T_cam, testPose, initEndEffPoseStamped_actual)
        T_offset = adpt_help.get_Tmat_TranlateInBodyF([0., 0., -10e-3]) # small offset from the target pose
        targetSearchPoseStamped_actual_Start = adpt_help.get_PoseStamped_from_T_initPose(T_offset, targetPoseStamped_actual)
        p_xyz_supposed2B = np.array([targetSearchPoseStamped_actual_Start.pose.position.x, targetSearchPoseStamped_actual_Start.pose.position.y, targetSearchPoseStamped_actual_Start.pose.position.z])
        q_xyzw_supposed2B = np.array([targetSearchPoseStamped_actual_Start.pose.orientation.x, targetSearchPoseStamped_actual_Start.pose.orientation.y, targetSearchPoseStamped_actual_Start.pose.orientation.z,targetSearchPoseStamped_actual_Start.pose.orientation.w])
        r_supposed = Rot.from_quat(q_xyzw_supposed2B)
        

        r_diff = r_supposed.inv() * r_measured
        mrp_diff = r_diff.as_mrp()
        thetaDiff = np.arctan(np.linalg.norm(mrp_diff))*4 * 180/np.pi
        positionDiff = p_xyz_supposed2B-p_xyz_measured

        positionDiff_list.append(np.linalg.norm(positionDiff))
        AngleDiff_list.append(thetaDiff)
    
    validIndex = np.where( np.logical_and(np.array(positionDiff_list)<2e-3, np.array(AngleDiff_list)<2) )
    
    if (validIndex == thisPoseIdx).any():
        return thisPoseIdx
    else:        
        closestIdx = np.argmin(np.abs(validIndex[0]-(thisPoseIdx+prevCorrection)))
        print(validIndex, thisPoseIdx, closestIdx)
        return validIndex[0][closestIdx]
        
        

#===============================================================================================================

def main(args):
    if args.plot3D:
        from visualization import Visualizer3D as vis3d

    thisFolder = args.folderDateName
    radius = args.radius
    imgSize = args.imgSize


    adpt_help = adaptMotionHelp()
    expDataFolder = Path.joinpath(Path.home(), 'TaeExperiment')
    thisMatFolderPath = str(Path.joinpath(expDataFolder, thisFolder))
    thisGraspFolderPath = str(Path.joinpath(expDataFolder, 'tmpPlannedGrasp'))

    fileList = []
    prevCorrection = 0
    for file in os.listdir(thisMatFolderPath): 
        if file.endswith("_NON.mat"):        
            
            matData = loadmat(os.path.join(thisMatFolderPath, file))
            correspondingGraspFolder = matData['storedDataDirectory'][0]
            correspondingGraspFolder = os.path.normpath(correspondingGraspFolder)
            correspondingGraspFolderName = correspondingGraspFolder.split(os.sep)[-2:]
            
            graspFolder = os.path.join(thisGraspFolderPath,correspondingGraspFolderName[0], correspondingGraspFolderName[1])
            with open(os.path.join(graspFolder, 'gqcnnResult.p'), 'rb') as handle:
                gqcnn_data = pickle.load(handle)
            with open(os.path.join(graspFolder, 'point_normalResult.p'), 'rb') as handle:
                surfaceNorm_data = pickle.load(handle)
            with open(os.path.join( graspFolder, 'TransformMat_board_verified'), 'rb') as handle:
                T_cam = pickle.load(handle)

            # Find this pose    
            GQCNN_poses = gqcnn_data['poses']
            pntNorml_poses = surfaceNorm_data['selectedPoint_normlVect']
            poseList = GQCNN_poses + pntNorml_poses
            
            thisPoseIdx_saved = matData['thisPoseIdx'][0][0]            
            thisPoseIdx=checkActualPoseIdx(thisPoseIdx_saved, matData,poseList,adpt_help,T_cam, prevCorrection)
            prevCorrection = thisPoseIdx-thisPoseIdx_saved

            print(file, thisPoseIdx_saved, thisPoseIdx)
            thisPose = poseList[thisPoseIdx]

            # Get the Pointcloud        
            # pointCloudObj = surfaceNorm_data['pointCloudObj_filtered']
            # pointCloudCoordArrays = pointCloudObj.point_cloud.data.T
            pointCloudCoordArrays = surfaceNorm_data['pointCloud']

            # Visualize the point cloud
            if args.plot3D:
                vis3d.figure()        
                addedPoint = [thisPose.position.x, thisPose.position.y, thisPose.position.z]
                Rtemp = RFromq(np.array([thisPose.orientation.x,thisPose.orientation.y, thisPose.orientation.z, thisPose.orientation.w]))
                vect = -Rtemp[:,0] # Flip the sign because it is already flipped

                vis3d.arrow(addedPoint, vect/60, tube_radius=0.5e-3, color = (1.0, 0, 0))
                vis3d.points(pointCloudCoordArrays, scale=0.0005)
                vis3d.show(block=False) 
                
            


            # Orientation Transform of the point cloude to align image to suction axis
            setOrientation = tf.transformations.quaternion_from_euler(np.pi,0,np.pi/2,'sxyz') #static (s) rotating (r)
            initEndEffPoseStamped = PoseStamped()                  
            initEndEffPoseStamped.pose.orientation.x = setOrientation[0]
            initEndEffPoseStamped.pose.orientation.y = setOrientation[1]
            initEndEffPoseStamped.pose.orientation.z = setOrientation[2]
            initEndEffPoseStamped.pose.orientation.w = setOrientation[3]

            targetPoseStamped = adpt_help.getGoalPosestampedFromGQCNN(T_cam, thisPose, initEndEffPoseStamped) 

            

            # Make the current point an origin and rotate the point cloud 
            pointCloudArray_noOffset = pointCloudCoordArrays-np.array([[thisPose.position.x, thisPose.position.y, thisPose.position.z]])
            R_N_cam = T_cam[0:3,0:3]
            PC_noOffset_in_N = (R_N_cam@pointCloudArray_noOffset.T).T
            
            R_N_endEFF = RFromq(np.array([targetPoseStamped.pose.orientation.x,targetPoseStamped.pose.orientation.y,targetPoseStamped.pose.orientation.z,targetPoseStamped.pose.orientation.w]))



            # R_measured = RFromq(q_xyzw)
            # print(np.dot(R_N_endEFF[:,2], R_measured[:,2]))
            # if np.dot(R_N_endEFF[:,2], R_measured[:,2]) < 0.9:
            #     print('stopHere')

            PC_noOffset_in_endEff = (R_N_endEFF.T @ PC_noOffset_in_N.T).T

            

            validIdx = np.logical_and(np.abs(PC_noOffset_in_endEff[:,0])<radius, np.abs(PC_noOffset_in_endEff[:,1])<radius)
            valid_PC_noffset_in_endEff = PC_noOffset_in_endEff[validIdx,:]

            # Visualize the rotated point cloud
            if args.plot3D:
                vis3d.figure() 
                vis3d.arrow(np.array([0,0,0]), np.array([1/70.0,0,0]), tube_radius=0.5e-3, color = (1.0, 0, 0))
                vis3d.arrow(np.array([0,0,0]), np.array([0,1/70.0,0]), tube_radius=0.5e-3, color = (0, 1.0, 0))
                vis3d.arrow(np.array([0,0,0]), np.array([0,0,1/70.0]), tube_radius=0.5e-3, color = (0, 0, 1.0))
                vis3d.points(valid_PC_noffset_in_endEff, scale=0.0005)
                vis3d.show(block=False) 

            
            pcImg = point_cloud_to_image(valid_PC_noffset_in_endEff, imgSize, radius=radius, depthRange = 1e-2)
            grayImg = GrayscaleImage(pcImg)  
            grayImg.savefig(thisMatFolderPath, file[0:-4])         
            grayImg.save(os.path.join(thisMatFolderPath, file[0:-4]+'.npy'))        
            
            savingDictionary = {}
            savingDictionary['imgData'] = grayImg.data
            savingDictionary['imgPixelSize'] = imgSize
            savingDictionary['poseIdx'] = thisPoseIdx
            savemat(os.path.join(thisMatFolderPath, file[0:-4]+'_engageImg.mat'), savingDictionary)
            
            # input("Enter to continue")
    # fileList.append(os.path.join("/tmp", file))  

 

if __name__ == '__main__':  
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--plot3D', type=bool, help='PlotImg', default=False)
  parser.add_argument('--imgSize', type=int, help='ImgSize', default=32)
  parser.add_argument('--radius', type=float, help='ImgSize', default=2e-2)
  parser.add_argument('--folderDateName', type=str, help='savingFolderDateName', default="221212")
  
  

  args = parser.parse_args()    
  
  main(args)
  # main(depth, rotAngleList[mode], translateZList[mode])

