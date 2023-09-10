import numpy as np
import cv2
import cv2.aruco as aruco
import os


def loadAugImages(path):
    # """
    # :param path: folder in which all the marker images with ids are stored
    # :return: dictionary with key as the id and values as the augment image
    # """
    
    myList = os.listdir(path)
    noOfMarkers = len(myList)
    print("Total Number of Markers Detected:", noOfMarkers)
    augDic = {}
    for imgPath in myList:
        key = int(os.path.splitext(imgPath)[0])
        imgAug = cv2.imread(f'{path}/{imgPath}')
        augDic[key] = imgAug
    return augDic

def findArucoMarkers(img, markerSize=4, totalMarkers=50, draw=True):
    # """
    # :param img: iage in which to find the aruco markers
    # :param markerSize: the size of the markers
    # :param totalMarkers: total number of markers that composethe dictionary
    # :param draw: flag to draw bbox around markers detected
    # :return: bounding boes and id numbers of markers detected
    # """
    
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key = getattr(aruco,f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = aruco.Dictionary_get(key)
    arucoParam = aruco.DetectorParameters_create()
    bboxs, ids, rejected = aruco.detectMarkers(imgGray, arucoDict, parameters=arucoParam)
    #print(ids)
    
    if draw:
        aruco.drawDetectedMarkers(img, bboxs)
        
    return [bboxs, ids]

def augmentAruco(bbox, id, img, imgAug, drawId=True):
    # """
    # :param bbox: the four corner points of the box
    # :param id: marker id of the corresponding box used only for display
    # :param img: the final image on which to draw
    # :param imgAug: the image that will be overlapped on the marker
    # :param drawId: flag to display the id of the detected markers
    # :return: image with the augment image overlaid
    # """
    
    tl = int(bbox[0][0][0]), int(bbox[0][0][1])
    tr = bbox[0][1][0], bbox[0][1][1]
    br = bbox[0][2][0], bbox[0][2][1]
    bl = bbox[0][3][0], bbox[0][3][1]
    
    h, w, c = imgAug.shape
    
    pts1 = np.array([tl, tr, br, bl])
    pts2 = np.float32([[0,0],[w,0],[w,h],[0,h]])
    matrix, _ = cv2.findHomography(pts2, pts1)
    imgOut = cv2.warpPerspective(imgAug, matrix, (img.shape[1], img.shape[0]))
    cv2.fillConvexPoly(img, pts1.astype(int), (0,0,0))
    imgOut = img + imgOut
    
    if drawId:
        cv2.putText(imgOut, str(id), tl, cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 2)
    
    return imgOut

def augmentArucoBasic(bbox, id, img, drawId=True):    
    tl = int(bbox[0][0][0]), int(bbox[0][0][1])
    tr = bbox[0][1][0], bbox[0][1][1]
    br = bbox[0][2][0], bbox[0][2][1]
    bl = bbox[0][3][0], bbox[0][3][1]
    if drawId:
        cv2.putText(img, str(id), tl, cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 2)
    

def main():
    cap = cv2.VideoCapture(0)
    # imgAug = cv2.imread("Markers/23.jpg")
    augDic = loadAugImages("Markers")
    

    while True:
        success, img = cap.read()
        arucoFound = findArucoMarkers(img)
        
        # Loop through all the markers and augment each one
        if len(arucoFound[0])!=0:
            for bbox, id in zip(arucoFound[0], arucoFound[1]):
                # print(id)
                # img = augmentAruco(bbox, id, img, imgAug)
                if int(id) in augDic.keys():
                    img = augmentAruco(bbox, id, img, augDic[int(id)])
        
        cv2.imshow("Image", img)
        cv2.waitKey(1) #gives delay of 1 milisecond

if __name__ == "__main__":
    main()