# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 17:41:01 2017

This pipeline takes input image, used K-means clustering to group
like colors together, and then thresholds such that it will highlight the road

Python 3.5, OpenCV3.1, NumPy 1.13.0

@author: Frank Tranghese
Boston University College of Engineering
EC601 - Final Project - Seamless Track Detection
"""

import cv2
import numpy as np

img = cv2.imread('043.png') #load image

Z = img.reshape((-1,3)) #reshape image

# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
K = 5
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))

cv2.imshow('res2',res2)

hsv_img = cv2.cvtColor(res2, cv2.COLOR_BGR2HSV) #convert to HSV

[h,s,v] = cv2.split(hsv_img) #split into HSV channels

[retval, thresh1] = cv2.threshold(v, 170, 255, cv2.THRESH_BINARY)
cv2.imshow('Binary Threshold',thresh1)

kernel = np.ones((5,5),np.uint8)
erode = cv2.erode(thresh1,kernel,iterations = 1)

cv2.imshow('Dilation',erode)

cv2.waitKey(0)