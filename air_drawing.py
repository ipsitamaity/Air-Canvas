import cv2
import numpy as np
import time
import os
import Hand_Tracking_module as htm


brushThickness = 15
eraserThickness = 50

folderPath = "header"
myList = os.listdir(folderPath)
print(myList)

overlaylist = [] #for storing our images

for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlaylist.append(image)
print(len(overlaylist))

header = overlaylist[0]

drawColor = (255, 0, 255)

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

detector = htm.handDetector(detectionCon=0.85)
xp,yp = 0,0

imgCanvas = np.zeros((720, 1280, 3),np.uint8)

while True:

    #1. Import images
    success, img = cap.read()

    #2. finding hand landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList)!= 0:
           #print(lmList)

           #tip of index and middle finger
       x1,y1 = lmList[8][1:]
       x2,y2 = lmList[12][1:]
           # which finger are up
       fingers = detector.fingersup()
           #print(fingers)

            # selection finger
       if fingers[1] and fingers[2]:
           xp, yp = 0, 0
           if y1 < 125:
               if 50 < x1 < 200:#clicking the first number
                   header = overlaylist[0]
               elif 350 < x1 < 450:
                   header = overlaylist[1]
                   drawColor = (0,0,0)
               elif 450 < x1 < 550:
                   header = overlaylist[2]
                   drawColor = (238,0,0)
               elif 550 < x1 < 650:
                   header = overlaylist[3]
                   drawColor = (0,0,255)
               elif 650 < x1 < 750:
                   header = overlaylist[4]
                   drawColor = (255,185,15)
               elif 950 < x1 < 1200:
                   header = overlaylist[5]
                   drawColor = (0,0,0)
           cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

           
       if fingers[1] and fingers[2] ==False:
           cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
           print("drawing mode")
           if xp==0 and yp==0:
               xp, yp = x1, y1
           if drawColor == (0,0,0):
               cv2.line(img, (xp,yp),(x1,y1),drawColor,eraserThickness)
               cv2.line(imgCanvas, (xp,yp),(x1,y1),drawColor,eraserThickness)
           else:

               cv2.line(img, (xp,yp),(x1,y1),drawColor,brushThickness)
               cv2.line(imgCanvas, (xp,yp),(x1,y1),drawColor,brushThickness)
           xp, yp = x1, y1
    imgGray = cv2.cvtColor(imgCanvas,cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 225, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img,imgCanvas)

           
    img[0:125,0:1280] = header
    img = cv2.addWeighted(img,0.5,imgCanvas,0.5,0)
    cv2.imshow("Image",img)
    cv2.imshow("Canvas",imgCanvas)
    if cv2.waitKey(10) == ord('q'):
       break
    
