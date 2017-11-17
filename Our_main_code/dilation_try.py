import cv2
import numpy as np
import matplotlib.image as mpimg

#cap = cv2.VideoCapture(0)
cap = cv2.imread("test_images/test1.jpg")
#cap = mpimg.imread(cap)

while(1):

   # _, frame = cap.read()
    hsv = cv2.cvtColor(cap, cv2.COLOR_BGR2HSV)
    frame=cap 
    lower_red = np.array([30,150,50])
    upper_red = np.array([100,255,180])
    
    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(frame,frame, mask= mask)

    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(mask,kernel,iterations = 1)
    dilation = cv2.dilate(mask,kernel,iterations = 1)

    cv2.imshow('Original',frame)
    cv2.imshow('Mask',mask)
    cv2.imshow('Erosion',erosion)
    cv2.imshow('Dilation',dilation)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
