#!/usr/bin/env python

import cv2
from cv2 import aruco
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

import glob
import random
import sys
import pickle

aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
# Note: Pattern generated using the following link
# https://calib.io/pages/camera-calibration-pattern-generator
# board = cv2.aruco.CharucoBoard_create(11, 8, 0.015, 0.011, aruco_dict)


def read_chessboards(frames):
    """
    Charuco base pose estimation.
    """
    all_corners = []
    all_ids = []

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict)

        if len(corners) > 0:
            ret, c_corners, c_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
            # ret is the number of detected corners
            if ret > 0:
                all_corners.append(c_corners)
                all_ids.append(c_ids)
        else:
            print('Failed!')

    imsize = gray.shape
    return all_corners, all_ids, imsize


def capture_camera(dev_num=0, num=1, mirror=False, size=None):
    frames = []

    cap = cv2.VideoCapture(dev_num)

    while True:
        ret, frame = cap.read()

        if mirror is True:
            frame = cv2.flip(frame, 1)

        if size is not None and len(size) == 2:
            frame = cv2.resize(frame, size)

        # My config applies floating layout for windows named 'Java'
        cv2.imshow('Java', frame)

        k = cv2.waitKey(1)
        if k == 27:  # Esc
            break
        elif k == 10 or k == 32:  # Enter or Space
            frames.append(frame)
            print('Frame captured!')
            if len(frames) == num:
                break

    return frames


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
    workdir = os.path.expanduser('~') + "/catkin_ws/src/tae_ur_experiment/src/Aruco/workdir"
    # workdir = "./workdir/"
    # aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    # aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

    board_eff = aruco.GridBoard_create(1,1,0.02,0.005,aruco_dict,firstMarker=2)
    imboard = board_eff.draw((4000, 4000))
    print workdir + "/board_eff.jpg"
    print cv2.imwrite(workdir + "/board_eff.tiff", imboard)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.imshow(imboard, cmap = mpl.cm.gray, interpolation = "nearest")
    ax.axis("off")
    plt.show(block=False)    
    plt.pause(1)

    # board_anchor = aruco.GridBoard_create(1,1,0.02,0.005,aruco_dict,firstMarker=20)
    board_anchor = aruco.GridBoard_create(3,3,0.01,0.005,aruco_dict,firstMarker=20)
    imboard = board_anchor.draw((4000, 4000))
    print workdir + "/board_marker.jpg"
    print cv2.imwrite(workdir + "/board_marker.tiff", imboard)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.imshow(imboard, cmap = mpl.cm.gray, interpolation = "nearest")
    ax.axis("off")
    plt.show(block=False)    
    plt.pause(1)


    board_tank = aruco.GridBoard_create(3,3,0.02,0.005,aruco_dict,firstMarker=7)
    # board = aruco.CharucoBoard_create(3, 3, 1, 0.6, aruco_dict)
    imboard = board_tank.draw((4000, 4000))
    print workdir + "/board_tank.jpg"
    print cv2.imwrite(workdir + "/board_tank.tiff", imboard)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.imshow(imboard, cmap = mpl.cm.gray, interpolation = "nearest")
    ax.axis("off")
    plt.show(block=False)    
    plt.pause(1)

    datadir = os.path.expanduser('~') + "/catkin_ws/src/tae_ur_experiment/src/Aruco/workdir/"
    
    # Load data (deserialize)
    with open(datadir + 'calibMtx.pickle', 'rb') as handle:
        loaded_data = pickle.load(handle)

    camera_matrix = loaded_data['mtx']
    dist_coeff = loaded_data['dist']

    
    # Real-time axis drawing
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        
        k = cv2.waitKey(1)
        if k == 27:  # Esc
            break
        output = draw_axis(frame, camera_matrix, dist_coeff, board_tank, False)
        output = draw_axis(frame, camera_matrix, dist_coeff, board_anchor, False)
        output = draw_axis(frame, camera_matrix, dist_coeff, board_eff, False)
        
        frame = cv2.undistort(src = frame, cameraMatrix = camera_matrix, distCoeffs = dist_coeff)
        cv2.imshow('Java', frame)

        


if __name__ == '__main__':
    main()