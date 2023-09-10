import numpy as np
import cv2, PIL, os
from cv2 import aruco
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
# %matplotlib nbagg
import pickle



workdir = os.path.expanduser('~') + "/catkin_ws/src/tae_ur_experiment/src/Aruco/workdir/"
datadir = workdir


# Load data (deserialize)
with open(datadir + 'calibMtx.pickle', 'rb') as handle:
    loaded_data = pickle.load(handle)

mtx = loaded_data['mtx']
dist = loaded_data['dist']
print mtx



images = np.array([datadir + f for f in os.listdir(datadir) if f.endswith(".jpg") ])

i=30 # select image id
plt.figure()
frame = cv2.imread(images[i])
img_undist = cv2.undistort(frame,mtx,dist,None)
plt.subplot(1,2,1)
plt.imshow(frame)
plt.title("Raw image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(img_undist)
plt.title("Corrected image")
plt.axis("off")
plt.show()