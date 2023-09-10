#!/usr/bin/env python

import os, sys
import rospy

import numpy as np
import cv2
import cv2.aruco as aruco
import os
import pickle

from datetime import datetime

from tae_psoc.msg import SensorPacket
from tae_psoc.msg import cmdToPsoc

IDLE = 0
STREAMING = 1


NO_CMD = 0
START_CMD = 2
IDLE_CMD = 3
RECORD_CMD = 10

currState = IDLE
CMD_in = NO_CMD

aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)

def callback(data):
    global CMD_in        
    CMD_in = data.cmdInput
    print CMD_in

# def loadAugImages(path):
#     # """
#     # :param path: folder in which all the marker images with ids are stored
#     # :return: dictionary with key as the id and values as the augment image
#     # """
    
#     myList = os.listdir(path)
#     noOfMarkers = len(myList)
#     print("Total Number of Markers Detected:", noOfMarkers)
#     augDic = {}
#     for imgPath in myList:
#         key = int(os.path.splitext(imgPath)[0])
#         imgAug = cv2.imread(f'{path}/{imgPath}')
#         augDic[key] = imgAug
#     return augDic

def findArucoMarkers(img, markerSize=4, totalMarkers=50, draw=True):
    # """
    # :param img: iage in which to find the aruco markers
    # :param markerSize: the size of the markers
    # :param totalMarkers: total number of markers that composethe dictionary
    # :param draw: flag to draw bbox around markers detected
    # :return: bounding boes and id numbers of markers detected
    # """
    
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key = getattr(aruco,'DICT_4X4_50') # For now let's force the marker key.
    # key = getattr(aruco,f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = aruco.Dictionary_get(key)
    arucoParam = aruco.DetectorParameters_create()
    bboxs, ids, rejected = aruco.detectMarkers(imgGray, arucoDict, parameters=arucoParam)
    #print(ids)
    
    if draw:
        aruco.drawDetectedMarkers(img, bboxs)
        
    return [bboxs, ids]

def augmentArucoBasic(bbox, id, img, drawId=True):    
    tl = int(bbox[0][0][0]), int(bbox[0][0][1])
    tr = bbox[0][1][0], bbox[0][1][1]
    br = bbox[0][2][0], bbox[0][2][1]
    bl = bbox[0][3][0], bbox[0][3][1]
    if drawId:
        cv2.putText(img, str(id), tl, cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 2)

def draw_axis(frame, camera_matrix, dist_coeff, board, verbose=True):
    corners, ids, rejected_points = cv2.aruco.detectMarkers(frame, aruco_dict)
    
    if corners is None or ids is None:
        return None
    if len(corners) != len(ids) or len(corners) == 0:
        return None

    try:
        ret, p_rvec,p_tvec = aruco.estimatePoseBoard(corners,ids, board, camera_matrix, dist_coeff) 

        
        # ret, p_rvec, p_tvec = cv2.aruco.estimatePoseCharucoBoard(c_corners,
        #                                                         c_ids,
        #                                                         board,
        #                                                         camera_matrix,
        #                                                         dist_coeff)
        if p_rvec is None or p_tvec is None:
            return None
        if np.isnan(p_rvec).any() or np.isnan(p_tvec).any():
            return None
        cv2.aruco.drawAxis(frame,
                        camera_matrix,
                        dist_coeff,
                        p_rvec,
                        p_tvec,
                        0.02)
        # cv2.aruco.drawDetectedCornersCharuco(frame, c_corners, c_ids)
        # cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        # cv2.aruco.drawDetectedMarkers(frame, rejected_points, borderColor=(100, 0, 240))
    except cv2.error:
        return None

    if verbose:
        print('Translation : {0}'.format(p_tvec))
        print('Rotation    : {0}'.format(p_rvec))
        print('Distance from camera: {0} m'.format(np.linalg.norm(p_tvec)))

    return [p_rvec, p_tvec]    

def main():
    global currState
    global CMD_in  
    
    size_of_marker =  0.020 # side lenght of the marker in meter    
    # size_of_marker =  0.01 # side lenght of the marker in meter    

    datadir = os.path.expanduser('~') + "/catkin_ws/src/tae_ur_experiment/src/Aruco/workdir/"
    
    # Load data (deserialize)
    with open(datadir + 'calibMtx.pickle', 'rb') as handle:
        loaded_data = pickle.load(handle)

    mtx = loaded_data['mtx']
    dist = loaded_data['dist']
    print mtx

    
    print "here"
    # imgAug = cv2.imread("Markers/23.jpg")
    # augDic = loadAugImages("Markers")

    rospy.init_node('tae_trackWheel')
    pub = rospy.Publisher('SensorPacket', SensorPacket, queue_size=10)
    rospy.Subscriber("cmdToPsoc",cmdToPsoc, callback)
    msg = SensorPacket()

    #Set the boards and save them. 
    board = []    
    # board.append(aruco.GridBoard_create(1,1,0.02,0.005,aruco_dict,firstMarker=0)) #Anchor Marker single
    board.append(aruco.GridBoard_create(1,1,0.02,0.005,aruco_dict,firstMarker=20)) # single marker for flat plate
    # board.append(aruco.GridBoard_create(3,3,0.01,0.005,aruco_dict,firstMarker=20)) #Anchor Marker board
    board.append(aruco.GridBoard_create(3,3,0.02,0.005,aruco_dict,firstMarker=7)) #Tank
    board.append(aruco.GridBoard_create(1,1,0.02,0.005,aruco_dict,firstMarker=2)) #End-effector
 


    while not rospy.is_shutdown():
        
            if currState == IDLE and CMD_in == START_CMD:
            # if True:
                CMD_in = NO_CMD
                currState = STREAMING     
                cap = cv2.VideoCapture(0)
                 # We need to set resolutions.
                # so, convert them from float to integer.
                frame_width = int(cap.get(3))
                frame_height = int(cap.get(4))
                
                size = (frame_width, frame_height)
                
                # Below VideoWriter object will create
                # a frame of above defined The output 
                # is stored in 'filename.avi' file.
                ResultSavingDirectory = os.path.expanduser('~') + '/TaeExperiment/' + datetime.now().strftime("%y%m%d")
                if not os.path.exists(ResultSavingDirectory):
                    os.makedirs(ResultSavingDirectory)
                result = cv2.VideoWriter(ResultSavingDirectory + '/tmpFile.avi',cv2.VideoWriter_fourcc(*'MJPG'),30, size)

                recordFlag = False

                while not CMD_in == IDLE_CMD and not rospy.is_shutdown():
                    try:
                        if CMD_in == RECORD_CMD:
                            recordFlag = True
                            CMD_in = NO_CMD

                        success, frame = cap.read()

                        numMarkersGroup = 3

                        tvecs = np.zeros((numMarkersGroup,3))
                        rvecs = np.zeros((numMarkersGroup,3))

                        for i in range(0,numMarkersGroup):
                            output = draw_axis(frame, mtx, dist, board[i], False)
                            if output is not None:                                
                                rvecs[i,:] = output[0].reshape(3)
                                tvecs[i,:] = output[1].reshape(3)
                        
                        frame = cv2.undistort(src = frame, cameraMatrix = mtx, distCoeffs = dist)
                        
                        msg.pSensor_Pa = np.concatenate((np.reshape(tvecs, (1,3*numMarkersGroup)), np.reshape(rvecs, (1,3*numMarkersGroup)) ), axis=1)[0,:]  
                        msg.header.stamp = rospy.Time.now()  
                        pub.publish(msg)   

                        cv2.imshow("Image", frame)
                        cv2.waitKey(1) #gives delay of 1 milisecond
                        #Checking for commit 

                        if recordFlag:
                            result.write(frame)
                        
                    except Exception as e:
                     print "SensorComError: " + str(e)
                     pass
            
                cap.release()
                result.release()
                cv2.destroyAllWindows()

                CMD_in = NO_CMD
                currState = IDLE


       

if __name__ == "__main__":
    main()
