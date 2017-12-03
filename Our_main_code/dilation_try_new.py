import cv2
import numpy as np
import matplotlib.image as mpimg
import statistics
from functools import reduce
from math import sqrt

def stddev(lst):
    mean = float(sum(lst)) / len(lst)
    return sqrt(float(reduce(lambda x, y: x + y, map(lambda x: (x - mean) ** 2, lst))) / len(lst)) 

cap = cv2.imread("test_images/test4.jpg")
setmin = 0
change = 150
#varibles for a pixel picked from middle of frame
midpixx = 660 
midpixy = 540
shot = 0   
while(1):

   # _, frame = cap.read()
    hsv = cv2.cvtColor(cap, cv2.COLOR_BGR2HSV)
    frame=cap 
   # lower_red = np.array([1,150,1])
   # upper_red = np.array([255,255,255])
   # lower_red = np.array([0,10,80])
   # upper_red = np.array([255,255,2155])
   # lower_red = np.array([85,0,45])
   # upper_red = np.array([165,79,125])
   # lower_red = np.array([85,0,45])
   # upper_red = np.array([165,79,125])
   #[126  32  40] +/- 40 worked 
   #113  97  42
    #[125  39  85]
    #[138  31  83]
    H = []
    S = []
    V = []
    for nums in range(0,50):
   #get 50 pixels in each direction append values to individual lists
      up = hsv[(midpixx + shot ), (midpixy + shot)]
      H.append(up[0])
      S.append(up[1])
      V.append(up[2])
      down = hsv[(midpixx - shot ), (midpixy - shot)]
      H.append(down[0])
      S.append(down[1])
      V.append(down[2])
      mix1 = hsv[(midpixx + shot ), (midpixy - shot)]
      H.append(mix1[0])
      S.append(mix1[1])
      V.append(mix1[2])
      mix2 = hsv[(midpixx - shot ), (midpixy + shot)]
      H.append(mix2[0])
      S.append(mix2[1])
      V.append(mix2[2])
      shot += 1
    shot = 0
    #average and standard deviation for H S V
    Have = int(sum(H) / (len(H)))
    Save = int(sum(S) / (len(S)))
    Vave = int(sum(V) / (len(V))) 
    stdH = int(stddev(H))
    stdS = int(stddev(S))
    stdV = int(stddev(H))
   # print(Have, Save, Vave)
   # print(int(stddev(H)))
    lower_red = np.array([(Have-stdH),(Save-stdS),(Vave-stdV)])
    upper_red = np.array([(Have+stdH),(Save+stdS),(Vave+stdV)]) 
    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(frame,frame, mask= mask)
    
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(mask,kernel,iterations = 1)
    dilation = cv2.dilate(mask,kernel,iterations = 1)
    cv2.imwrite("show_mask.jpg",mask)
    cv2.imshow('Original',frame)
    cv2.imshow('Mask',mask)
    cv2.imshow('Erosion',erosion)
    cv2.imshow('Dilation',dilation)
   # if (change<setmax):
   #   change +=1
    if (change>setmin):
      change -=1
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()




