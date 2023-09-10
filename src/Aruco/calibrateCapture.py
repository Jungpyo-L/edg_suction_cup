import numpy as np
import cv2, PIL, os
from cv2 import aruco
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
# %matplotlib nbagg


# Make the ChAruco
workdir = os.path.expanduser('~') + "/catkin_ws/src/tae_ur_experiment/src/Aruco/workdir"
# workdir = "./workdir/"
# aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

singleMarker = aruco.drawMarker(aruco_dict,1,10)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.imshow(singleMarker, cmap = mpl.cm.gray, interpolation = "nearest")
ax.axis("off")
plt.show()

board = aruco.CharucoBoard_create(7, 5, 1, .8, aruco_dict)
imboard = board.draw((2000, 2000))
print workdir + "/chessboard.jpg"
print cv2.imwrite(workdir + "/chessboard.tiff", imboard)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.imshow(imboard, cmap = mpl.cm.gray, interpolation = "nearest")
ax.axis("off")
plt.show()


# Capture multiple images
imagesFolder = workdir
cap = cv2.VideoCapture(0)
count = 0
while(count < 50):    
    
    ret, frame = cap.read()

    cv2.imshow("Image", frame)
    # cv2.waitKey(1) #gives delay of 1 milisecond
    k = cv2.waitKey(100)
    if k == 99: # "c"
        filename = imagesFolder + "/image_" +  str(int(count)) + ".jpg"
        count += 1    
        print count
        print cv2.imwrite(filename, frame)    
cap.release()
print ("Done!")

