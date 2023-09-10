import numpy as np
import cv2
import cv2.aruco as aruco
import os
import ArucoModule as arm


def calcAngles(arucoFound):
    m = np.array([0,1])
    if len(arucoFound[1]) > 3:
        print('OH NO! Too many markers!')
        print(len(arucoFound[1]))
    elif len(arucoFound[1]) < 3:
        print('OH NO! Too few markers!')
        print(len(arucoFound[1]))
        
    bboxs = arucoFound[0]
    id = arucoFound[1]
    id_2 = []
    id_ind = []
    x = 0
    for i in id:
        id_2.append(i[0])
        id_ind.append(x)
        x += 1
        
    p0 = bboxs[id_2.index(m[0])][0][0]
    p1 = bboxs[id_2.index(m[1])][0][0]
    
    id_ind.remove(id_2.index(m[0]))
    id_ind.remove(id_2.index(m[1]))
    px = bboxs[id_ind[0]][0][0]
    
    vector_1 = p1-p0
    vector_2 = px-p1
    
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)

    print("{:.2f} degrees".format(angle*180/np.pi))
    return angle*180/np.pi
    
def mousePoints(event,x,y,flags,params):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img,(x,y),5,(0,0,255),cv2.FILLED)
        pointsList.append([x,y])
        print(pointsList)

def main():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    pointsList = []
    angles = []
    while True:
        success, img = cap.read()
        arucoFound = arm.findArucoMarkers(img)
        # Loop through all the markers and augment each one
        if len(arucoFound[0])!=0:
            for bbox, id in zip(arucoFound[0], arucoFound[1]):
                arm.augmentArucoBasic(bbox, id, img)
        
        cv2.imshow("Image", img)
        # cv2.setMouseCallback("Image", mousePoints)
        # cv2.waitKey(1) #gives delay of 1 milisecond
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # pointsList = []
            angles.append(calcAngles(arucoFound))
        if len(angles) == 2:
            print("Bar moved {:.2f} degrees!".format(angles[1]-angles[0]))
            angles = []
        
        # input("Press Enter to display angle...")

if __name__ == "__main__":
    main()