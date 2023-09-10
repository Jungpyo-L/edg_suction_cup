#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright ©2017. The Regents of the University of California (Regents).
All Rights Reserved. Permission to use, copy, modify, and distribute this
software and its documentation for educational, research, and not-for-profit
purposes, without fee and without a signed licensing agreement, is hereby
granted, provided that the above copyright notice, this paragraph and the
following two paragraphs appear in all copies, modifications, and
distributions. Contact The Office of Technology Licensing, UC Berkeley, 2150
Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-7201,
otl@berkeley.edu,
http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.

IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF
THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED
HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE
MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

Displays robust grasps planned using a FC-GQ-CNN-based policy on a set of saved
RGB-D images.

Author
------
Mike Danielczuk, Jeff Mahler
"""

import argparse
import numpy as np
import os, sys
import rospy
import matplotlib as mpl
# mpl.use("TkAgg")
# mpl.use("Agg")
import matplotlib.pyplot as plt
from timeit import default_timer as timer

from scipy.spatial.transform import Rotation as R

from autolab_core import Logger, Point, CameraIntrinsics, DepthImage, BinaryImage
from visualization import Visualizer2D as vis2d
from visualization import Visualizer3D as vis3d

from std_msgs.msg import Header
from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import Pose

from gqcnn_ros.msg import GQCNNGrasp_multiple
from std_msgs.msg import Int16


import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pyrealsense2 as rs
import time
from datetime import datetime
import pickle
# from perception.perception.rgbd_sensors import RgbdSensorFactory
from perception import RgbdSensorFactory

from catkin.workspace import get_source_paths, get_workspaces
print(get_workspaces())
datadir = os.path.dirname(os.path.realpath(__file__))
from .kinect_smoothing import Denoising_Filter




class GraspProcessor(object):
    def __init__(self,          
                 gripper_width=0.05,
                 vis_grasp=True,
                 depth_thres = 0.29,
                 num_actions = 10):
        # self.depth_im = depth_im
        # self.camera_intr = camera_intr
        self.gripper_width = gripper_width
        self.vis_grasp = vis_grasp

        self.depth_im = None
        self.cur_q_val = None
        self.grasp_req_times = []
        self.grasp_plan_times = []

        self.graspPointObtained = False
        self.plannedGrasps = []
        self.plannedGraspsQvals=[]

        self.depth_thres = depth_thres
        self.numAction = num_actions
        
        # Initialize the ROS node.
        # rospy.init_node("gqcnn_ros_client")
        self.actionNum_pub = rospy.Publisher('actionNum', Int16, queue_size=1)
        self.cam_info_pub = rospy.Publisher(rospy.resolve_name("~camera_info",
                                                        "/gqcnn"),
                                    CameraInfo,
                                    queue_size=10)

        # Read image and set up publisher
            # depth_im = DepthImage.open(depth_im_filename, frame=camera_intr.frame)
        self.depth_pub = rospy.Publisher(rospy.resolve_name("~image", "/gqcnn"),
                                    Image,
                                    queue_size=10)
        self.seg_pub = rospy.Publisher(rospy.resolve_name("~mask", "/gqcnn"),
                                Image,
                                queue_size=10)          
        self.grasp_sub = rospy.Subscriber(rospy.resolve_name("~grasp", "/gqcnn"),
                                    GQCNNGrasp_multiple, self.process)                                                

        # Do not know why but it does not publish the action # at once.        
        rospy.sleep(0.1)
        for i in range(5):
            self.actionNum_pub.publish(self.numAction)
            rospy.sleep(0.1)


    @property
    def request_time_statistics(self):
        return len(self.grasp_req_times), np.mean(
            self.grasp_req_times), np.std(self.grasp_req_times)

    @property
    def planning_time_statistics(self):
        return len(self.grasp_plan_times), np.mean(
            self.grasp_plan_times), np.std(self.grasp_plan_times)

    def record_request_start(self):
        self.grasp_start_time = timer()

    def record_request_end(self):
        self.grasp_req_times.append(timer() - self.grasp_start_time)

    def record_plan_time(self, plan_time):
        self.grasp_plan_times.append(plan_time)

    def process(self, grasp):
        self.record_request_end()
        self.record_plan_time(grasp.plan_time)
        
        decendingOrder = np.flip(np.argsort(grasp.q_value))
        saveCounter = 0
        for i in decendingOrder:
            self.plannedGraspsQvals.append(grasp.q_value[i])
            self.plannedGrasps.append(grasp.pose[i])
            vis2d.figure(size=(5, 5))
            vis2d.imshow(self.depth_im, vmin=0.15, vmax=0.35)

            color = plt.get_cmap('hsv')(0.3 * grasp.q_value[i])[:-1]

            center_px = [grasp.center_0[i], grasp.center_1[i]]
            center = self.camera_intr.deproject_pixel(
                grasp.depth[i],
                Point(np.array(center_px), frame=self.camera_intr.frame))
            plt.scatter(*center_px, color=color, marker=".", s=100)          

            plt.title("Planned grasp on depth (Q=%.3f)" % (grasp.q_value[i]))
            plt.savefig(
                os.path.join(self.ResultSavingDirectory,
                            "grasp" +str(saveCounter)+".png"))            
            plt.close() 
            saveCounter+=1

        self.graspPointObtained = True

            
    def discover_cams(self):
        """Returns a list of the ids of all cameras connected via USB."""
        ctx = rs.context()
        ctx_devs = list(ctx.query_devices())
        ids = []
        for i in range(ctx.devices.size()):
            ids.append(ctx_devs[i].get_info(rs.camera_info.serial_number))
        return ids

    def getGQCNN_Grasps(self):
        if self.depth_im is None:
            self.graspPointObtained = False
            self.plannedGrasps = []
            self.plannedGraspsQvals= []
            ## =========== Read Camera Img =============== ##
            ids = self.discover_cams()
            assert ids, "[!] No camera detected."

            cfg = {}
            cfg["cam_id"] = ids[0]
            cfg["filter_depth"] = True
            cfg["frame"] = "realsense_overhead"

            sensor = RgbdSensorFactory.sensor("realsense", cfg)
            sensor.start()
            time.sleep(1) # needs some time to restart the camera.
            camera_intr = sensor.color_intrinsics
            
            # print(camera_intr)
            print("intrinsics matrix: {}".format(camera_intr.K))
            color_im, depth_im   = sensor.frames()
            sensor.stop()

            # print("intrinsics matrix: {}".format(camera_intr.K))

            _, axes = plt.subplots(1, 2)
            for ax, im in zip(axes, [color_im.data, depth_im.data]):
                ax.imshow(im)
                ax.axis("off")
            
            
            
            ResultSavingDirectory = os.path.join(os.path.expanduser('~'), "SuctionExperiment", "tmpPlannedGrasp", datetime.now().strftime("%y%m%d"), datetime.now().strftime("%H%M%S"))
            if not os.path.exists(ResultSavingDirectory):
                os.makedirs(ResultSavingDirectory)
            self.ResultSavingDirectory = ResultSavingDirectory

            plt.savefig(
                os.path.join(self.ResultSavingDirectory, "grasp_raw.png"))


            plt.show(block=False)
            # plt.ion()  
            # plt.show()
            plt.pause(1)
            plt.close()
            
            depth_im = depth_im.inpaint()

            self.color_im = color_im
            self.depth_im = depth_im
            self.camera_intr = camera_intr

        else:
            color_im = self.color_im
            depth_im = self.depth_im
            camera_intr = self.camera_intr



        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = camera_intr.frame            
            
        
        segmask = BinaryImage(np.iinfo(np.uint8).max *
                            (1 * (depth_im.data < self.depth_thres)).astype(np.uint8),
                            frame=depth_im.frame)

        segmask_msg = segmask.rosmsg
        segmask_msg.header = header
            # segmask = BinaryImage(np.iinfo(np.uint8).max *
            #                       np.ones(depth_im.shape).astype(np.uint8),
            #                       frame=depth_im.frame)   
        # Set up subscribers and processor

        
        self.applyfilter()
        depth_im_filtered = self.depth_im_filtered

        # depth_im_filtered = depth_im # if wants to ignore the filter


        camera_info_msg = CameraInfo()
        camera_info_msg.header = header
        camera_info_msg.K = np.array([
            camera_intr.fx, camera_intr.skew, camera_intr.cx, 0.0, camera_intr.fy,
            camera_intr.cy, 0.0, 0.0, 1.0
        ])

        depth_im_msg = depth_im_filtered.rosmsg
        depth_im_msg.header = header


        req_num = 0
        print("Start Publishing")
        rospy.sleep(1)

        self.record_request_start()
        self.cam_info_pub.publish(camera_info_msg)
        self.depth_pub.publish(depth_im_msg)
        self.seg_pub.publish(segmask_msg)
        
        while not rospy.is_shutdown() and not self.graspPointObtained:
            if self.request_time_statistics[0] > 0:
                rospy.loginfo(
                    "Request {:d} took {:.4f} s total ({:.4f} s planning)".format(
                        req_num - 1, self.grasp_req_times[-1],
                        self.grasp_plan_times[-1]))
            self.record_request_start()
            # self.cam_info_pub.publish(camera_info_msg)
            # self.depth_pub.publish(depth_im_msg)
            # self.seg_pub.publish(segmask_msg)
            req_num += 1
            rospy.sleep(3)
        
        # Save data into the folder
        data = dict()        
        data['color_im'] = color_im
        data['depth_im'] = depth_im
        data['depth_im_filtered'] = depth_im_filtered
        data['poses'] = self.plannedGrasps
        data['poses_qVal'] = self.plannedGraspsQvals
        data['camera_intr'] = self.camera_intr

        file = open(self.ResultSavingDirectory+'/gqcnnResult.p', 'wb')
        pickle.dump(data, file)
        file.close()



        rospy.loginfo("Request Times ({:d} trials): {:.4f} +- {:.4f} s".format(
            *self.request_time_statistics))
        rospy.loginfo("Planning Times ({:d} trials): {:.4f} +- {:.4f} s".format(
            *self.planning_time_statistics))

            

    def getSurfaceNormals_PointCloud(self, zMin=0.25, zMax = 0.32, disturbanceThres = 1e-3, checkRadius = 2e-3, 
                                     numberInRadiusThres=30, vectorNum = 50, visualizeOn = False):
        if self.depth_im is None:
            ## =========== Read Camera Img =============== ##
            ids = self.discover_cams()
            assert ids, "[!] No camera detected."

            cfg = {}
            cfg["cam_id"] = ids[0]
            cfg["filter_depth"] = True
            cfg["frame"] = "realsense_overhead"

            sensor = RgbdSensorFactory.sensor("realsense", cfg)
            sensor.start()
            time.sleep(1) # needs some time to restart the camera.
            camera_intr = sensor.color_intrinsics
            
            # print(camera_intr)
            print("intrinsics matrix: {}".format(camera_intr.K))
            color_im, depth_im   = sensor.frames()
            sensor.stop()

            print("intrinsics matrix: {}".format(camera_intr.K))

            _, axes = plt.subplots(1, 2)
            for ax, im in zip(axes, [color_im.data, depth_im.data]):
                ax.imshow(im)
                ax.axis("off")
                            
            ResultSavingDirectory = os.path.join(os.path.expanduser('~'), "SuctionExperiment", "tmpPlannedGrasp", datetime.now().strftime("%y%m%d"), datetime.now().strftime("%H%M%S"))
            if not os.path.exists(ResultSavingDirectory):
                os.makedirs(ResultSavingDirectory)
            self.ResultSavingDirectory = ResultSavingDirectory

            plt.savefig(
                os.path.join(ResultSavingDirectory, "grasp_raw.png"))


            # plt.show(block=False)    
            # plt.pause(1)
            # plt.close()

            
            depth_im = depth_im.inpaint()

            self.color_im = color_im
            self.depth_im = depth_im
            self.camera_intr = camera_intr
        
        else:
            color_im = self.color_im
            depth_im = self.depth_im
            camera_intr = self.camera_intr

        pointNormalCloud_raw = depth_im.point_normal_cloud(camera_intr)  # great if we to train for random normal vects.    
        pointCloud_raw = pointNormalCloud_raw.point_cloud.data.T


        self.applyfilter()
        depth_im_filtered = self.depth_im_filtered

        pointNormalCloud = depth_im_filtered.point_normal_cloud(camera_intr)  # great if we to train for random normal vects.    
        pointCloud = pointNormalCloud.point_cloud.data.T

        distVect = np.linalg.norm(pointCloud-pointCloud_raw,axis=1)
        dispCriteria = distVect < disturbanceThres

        validIdx = np.logical_and(pointCloud[:,2] < zMax, pointCloud[:,2] > zMin)
        validIdx = np.logical_and(validIdx, dispCriteria) # to filterout the edges

        
        validPointCloud = pointCloud[validIdx,:]
        validNormalVects = pointNormalCloud.normal_cloud.data.T[validIdx,:]

        if visualizeOn:
            vis3d.figure()
        
        randomI = np.random.permutation(len(validPointCloud))
        edgeRegionCrit = checkRadius                
        counter = 0
        
        point_normVect_list = []
        for idx in randomI:   
            addedPoint = validPointCloud[idx,:]
            numPointsInRegion = np.sum(np.linalg.norm(validPointCloud-addedPoint, axis=1) < edgeRegionCrit)
            # print(numPointsInRegion)
            if numPointsInRegion > numberInRadiusThres:
                vect = validNormalVects[idx,:]
                if visualizeOn:
                    vis3d.arrow(addedPoint, vect/60, tube_radius=0.5e-3, color = (1.0, 0, 0))
                
                # Save it to the pose stamp
                thisPose = Pose()
                thisPose.position.x = addedPoint[0]
                thisPose.position.y = addedPoint[1]
                thisPose.position.z = addedPoint[2]
                
                vect /= np.linalg.norm(vect)
                vect *= -1 #flip the sign
                # find the rotation that rotates x axis into the                 
                
                randY = np.random.rand(3)
                randY /= np.linalg.norm(randY)

                randZ = np.cross(vect, randY)
                randZ /= np.linalg.norm(randZ)

                randY = np.cross(randZ, vect)
                randY /= np.linalg.norm(randY)
                
                R_mat = np.concatenate((vect.reshape(3,1), randY.reshape(3,1),randZ.reshape(3,1)),axis =1 )
                
                
                r = R.from_matrix(R_mat)
                quat = r.as_quat()

                thisPose.orientation.x = quat[0]
                thisPose.orientation.y = quat[1]
                thisPose.orientation.z = quat[2]
                thisPose.orientation.w = quat[3]



                point_normVect_list.append(thisPose)
                counter+=1
                if counter > vectorNum:
                    break
        
        if visualizeOn:
            vis3d.points(validPointCloud, scale=0.0005)
            vis3d.show(block=False)



        self.point_normVect_list = point_normVect_list

        # Save data into the folder
        data = dict()        
        data['color_im'] = color_im
        data['depth_im'] = depth_im
        data['depth_im_filtered'] = depth_im_filtered        
        data['camera_intr'] = self.camera_intr
        data['pointCloud'] = validPointCloud # Coordinates
        data['normVect'] = validNormalVects
        
        data['pointCloudObj_raw'] = pointNormalCloud_raw #pointCloud Object
        data['pointCloudObj_filtered'] = pointNormalCloud #pointCloud Object


        


        data['selectedPoint_normlVect'] = self.point_normVect_list        

        file = open(self.ResultSavingDirectory+'/point_normalResult.p', 'wb')
        pickle.dump(data, file)
        file.close()



    def applyfilter(self):
        depth_im = self.depth_im
        camera_intr = self.camera_intr

        noise_filter = Denoising_Filter(flag='gaussian', ksize=9, sigma=0.5)
        # noise_filter = Denoising_Filter(flag='anisotropic', theta=60) 
        denoise_image_frame = noise_filter.smooth_image_frames([depth_im.data])

        # this one to filter
        self.depth_im_filtered = DepthImage(denoise_image_frame[0],frame=depth_im.frame)

        # this one to not filter
        # self.depth_im_filtered = self.depth_im

    
    def visualizeGrasp(self, zMin= 29e-3, zMax = 32e-3):
        camera_intr = self.camera_intr
        depth_im_filtered = self.depth_im_filtered

        pointCloud = camera_intr.deproject(depth_im_filtered).data.T
        validPointCloud = pointCloud[validIdx,:]

        validIdx = np.logical_and(pointCloud[:,2] < zMax, pointCloud[:,2] > zMin)

        vis3d.figure()
        vis3d.points(validPointCloud, scale=0.0005)

        for pose in self.plannedGrasps:
            addedPoint = [pose.position.x, pose.position.y, pose.position.z]
            r = R.from_quat([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
            rotMat = r.as_matrix()
            vect = rotMat[:,0]

            vis3d.arrow(addedPoint, -vect/50, tube_radius=1e-3, color = (1.0, 0, 0))

        # vis3d.pose() # to show the reference axis
        vis3d.show(block = False)

    def resetImg(self):
        self.depth_im = None

    def getGQCNN_Grasps_Bin(self, fileAppendix='', TopLeftPix=[10, 10], BottomRightPix=[300, 500]):                
        if self.depth_im is None:
        # if True:
            # print("im here")
            self.graspPointObtained = False
            self.plannedGrasps = []
            self.plannedGraspsQvals= []
            # print("self.graspPointObtained: ", self.graspPointObtained)

            ## =========== Read Camera Img =============== ##
            ids = self.discover_cams()
            assert ids, "[!] No camera detected."

            cfg = {}
            cfg["cam_id"] = ids[0]
            cfg["filter_depth"] = True
            cfg["frame"] = "realsense_overhead"

            sensor = RgbdSensorFactory.sensor("realsense", cfg)
            sensor.start()
            time.sleep(1) # needs some time to restart the camera.
            camera_intr = sensor.color_intrinsics
            
            # print(camera_intr)
            # print("intrinsics matrix: {}".format(camera_intr.K))
            color_im, depth_im   = sensor.frames()
            sensor.stop()

            # print("intrinsics matrix: {}".format(camera_intr.K))

            _, axes = plt.subplots(1, 2)
            for ax, im in zip(axes, [color_im.data, depth_im.data]):
                ax.imshow(im)
                ax.axis("off")
            
            ResultSavingDirectory = os.path.join(os.path.expanduser('~'), "SuctionExperiment", "tmpPlannedGrasp", datetime.now().strftime("%y%m%d"), datetime.now().strftime("%H%M%S")+fileAppendix)
            if not os.path.exists(ResultSavingDirectory):
                os.makedirs(ResultSavingDirectory)
            self.ResultSavingDirectory = ResultSavingDirectory

            plt.savefig(
                os.path.join(self.ResultSavingDirectory, "grasp_raw.png"))


            plt.show(block=False)    
            # plt.show()
            plt.pause(1)
            plt.close()
            
            depth_im = depth_im.inpaint()

            self.color_im = color_im
            self.depth_im = depth_im
            self.camera_intr = camera_intr

        else:
            color_im = self.color_im
            depth_im = self.depth_im
            camera_intr = self.camera_intr



        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = camera_intr.frame            

        
        BinMask = np.full(depth_im.data.shape, True)
        # TopLeftPix = [10, 10] # x, y 
        # BottomRightPix = [400, 300] # x, y
        for i in range(TopLeftPix[1], BottomRightPix[1]):
            for j in range(TopLeftPix[0],BottomRightPix[0]):
                    BinMask[i,j] = True

        segmask = BinaryImage(np.iinfo(np.uint8).max *
                    (1 * ( (depth_im.data < self.depth_thres) & BinMask)).astype(np.uint8) ,
                    frame=depth_im.frame)

        
        segmask_msg = segmask.rosmsg
        segmask_msg.header = header
            # segmask = BinaryImage(np.iinfo(np.uint8).max *
            #                       np.ones(depth_im.shape).astype(np.uint8),
            #                       frame=depth_im.frame)   
        # Set up subscribers and processor

        
        self.applyfilter()
        depth_im_filtered = self.depth_im_filtered

        # depth_im_filtered = depth_im # if wants to ignore the filter


        camera_info_msg = CameraInfo()
        camera_info_msg.header = header
        camera_info_msg.K = np.array([
            camera_intr.fx, camera_intr.skew, camera_intr.cx, 0.0, camera_intr.fy,
            camera_intr.cy, 0.0, 0.0, 1.0
        ])

        depth_im_msg = depth_im_filtered.rosmsg
        depth_im_msg.header = header


        req_num = 0

        print("Start Publishing")
        self.record_request_start()
        self.cam_info_pub.publish(camera_info_msg)
        self.depth_pub.publish(depth_im_msg)
        self.seg_pub.publish(segmask_msg)
        rospy.sleep(1)

        while not rospy.is_shutdown() and not self.graspPointObtained:
            if self.request_time_statistics[0] > 0:
                rospy.loginfo(
                    "Request {:d} took {:.4f} s total ({:.4f} s planning)".format(
                        req_num - 1, self.grasp_req_times[-1],
                        self.grasp_plan_times[-1]))
            # self.record_request_start()
            # self.cam_info_pub.publish(camera_info_msg)
            # self.depth_pub.publish(depth_im_msg)
            # self.seg_pub.publish(segmask_msg)
            # req_num += 1
            rospy.sleep(3)
        
        # Save data into the folder
        data = dict()        
        data['color_im'] = color_im
        data['depth_im'] = depth_im
        data['depth_im_filtered'] = depth_im_filtered
        data['poses'] = self.plannedGrasps
        data['poses_qVal'] = self.plannedGraspsQvals
        data['camera_intr'] = self.camera_intr

        file = open(self.ResultSavingDirectory+'/gqcnnResult.p', 'wb')
        pickle.dump(data, file)
        file.close()



        rospy.loginfo("Request Times ({:d} trials): {:.4f} +- {:.4f} s".format(
            *self.request_time_statistics))
        rospy.loginfo("Planning Times ({:d} trials): {:.4f} +- {:.4f} s".format(
            *self.planning_time_statistics))