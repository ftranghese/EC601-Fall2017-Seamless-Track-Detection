#This program trys to see the 
#the picked pixel value is 520 230
import cv2
import numpy as np
cap = cv2.imread("test_images/043.png")
hsv = cv2.cvtColor(cap, cv2.COLOR_BGR2HSV)



px = hsv[230,520]
print (px)

