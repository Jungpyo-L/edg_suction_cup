
import os
from pathlib import Path
from scipy.io import loadmat
import pickle

from autolab_core import Logger, Point, CameraIntrinsics, DepthImage, BinaryImage

testDataDirectory = os.path.dirname(os.path.realpath(__file__))
suctionCupFolder = Path(testDataDirectory).parent.parent
matFileDir = Path.joinpath(suctionCupFolder, 'Data','MatFileData')
plannedGraspDir = Path.joinpath(suctionCupFolder, 'Data','tmpPlannedGrasp')


thisFolder = '220928'

thisMatFolderPath = str(Path.joinpath(matFileDir,thisFolder))
thisGraspFolderPath = str(Path.joinpath(plannedGraspDir,thisFolder))

fileList = []
for file in os.listdir(thisMatFolderPath):
    if file.endswith("_initTouch.mat"):
        print(file)
        data = loadmat(os.path.join(thisMatFolderPath, file))
        correspondingGraspFolder = data['storedDataDirectory'][0]
        correspondingGraspFolderName = os.path.split(correspondingGraspFolder)[-1]
        graspFolder = os.path.join(thisGraspFolderPath,correspondingGraspFolderName)
        with open(os.path.join(graspFolder, 'point_normalResult.p'), 'rb') as handle:
            loaded_data = pickle.load(handle)
       

        

        print('hi')
        

        # fileList.append(os.path.join("/tmp", file))  

