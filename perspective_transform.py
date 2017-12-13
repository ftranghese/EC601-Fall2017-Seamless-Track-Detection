# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 12:55:51 2017
@author: Frank Tranghese
Boston University College of Engineering
EC601 - Seamless Track Detection Project
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from combined_thresh import combined_thresh
#from Line import Line

def perspective_transform(img):
    ''' updated perspective transform to
        automatically pick the transform values 
    '''
    
    img_size = (img.shape[1], img.shape[0])
    
    binary = img
    
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary[binary.shape[0]//2:,:], axis=0)
    
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[100:midpoint]) + 100
    rightx_base = np.argmax(histogram[midpoint:-100]) + midpoint
    
    ''' Estimates for where the vanishing point is. Assuming halfway up the image
        src is the rectangle defined by the bottom of where the left and right
        start. We then estimate the vaninishing point to be at the same point
        but halfway up the image and inward towards the midpoint by about half
    '''
    
    src = np.float32(
            [[leftx_base,img_size[1]],
            [rightx_base,img_size[1]],
            [leftx_base+((leftx_base - midpoint)//2),img_size[1]//2],
            [rightx_base-((midpoint - rightx_base)//2),img_size[1]//2]])
    
    dst = np.float32(
            [[leftx_base,img_size[1]],
            [rightx_base,img_size[1]],
            [leftx_base,0],
            [rightx_base,0]])
    
    m = cv2.getPerspectiveTransform(src, dst)
    m_inv = cv2.getPerspectiveTransform(dst, src)

    warped = cv2.warpPerspective(img, m, img_size, flags=cv2.INTER_LINEAR)
    unwarped = cv2.warpPerspective(warped, m_inv, (warped.shape[1], warped.shape[0]), flags=cv2.INTER_LINEAR)  # DEBUG
    mm = warped.shape
    
    kk = mm[0]*mm[1]*0.85
    kkk = str(kk)
    cv2.putText(img,kkk,(30,70),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
    #file11 = open('file1.txt','w')
    #file11.write(str(kk))
    #file11.close()

    return warped, unwarped, m, m_inv

if __name__ == '__main__':
    img_file = 'test_images/test5.jpg'

    with open('calibrate_camera.p', 'rb') as f:
        save_dict = pickle.load(f)
    mtx = save_dict['mtx']
    dist = save_dict['dist']

    img = mpimg.imread(img_file)
    img = cv2.undistort(img, mtx, dist, None, mtx)

    warped, unwarped, m, m_inv = perspective_transform(img)
    
    
    
   # cv2.putText(kk, (30,70), 0, 1, (0,0,0), 2, cv2.LINE_AA)
    plt.imshow(warped, cmap='gray', vmin=0, vmax=1)
    plt.show()

    plt.imshow(unwarped, cmap='gray', vmin=0, vmax=1)
    
plt.show()
