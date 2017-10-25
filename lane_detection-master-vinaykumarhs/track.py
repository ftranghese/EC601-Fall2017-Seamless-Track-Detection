#! /usr/bin/env python
# Queue implemented.
import os
import sys
import csv
import cv2
import glob
import numpy as np
from ParticleFilter import ParticleFilter

if __name__ == "__main__":

    cv2.namedWindow('Lane Markers')
    imgs = glob.glob("images/*.png")
    
    intercepts = []

    xl_int_pf=ParticleFilter(N=1000,x_range=(0,1500),sensor_err=1,par_std=100)
    xl_phs_pf=ParticleFilter(N=1000,x_range=(15,90),sensor_err=0.3,par_std=1)
    xr_int_pf=ParticleFilter(N=1000,x_range=(100,1800),sensor_err=1,par_std=100)
    xr_phs_pf=ParticleFilter(N=1000,x_range=(15,90),sensor_err=0.3,par_std=1)

    #tracking queues
    xl_int_q = [0]*15 
    xl_phs_q = [0]*15
    count = 0

    # imgs = ['images/mono_0000002062.png']
    for fname in imgs:
        # Load image and prepare output image
        orig_img = cv2.imread(fname)
        

        # Scale down the image - Just for better display.
        orig_height,orig_width=orig_img.shape[:2]
        # orig_img=cv2.resize(orig_img,(orig_width/2,orig_height/2),interpolation = cv2.INTER_CUBIC)
        # orig_height,orig_width=orig_img.shape[:2]

        # Part of the image to be considered for lane detection
        upper_threshold=0.4
        lower_threshold=0.2
        # Copy the part of original image to temporary image for analysis.
        img=orig_img[int(upper_threshold*orig_height):int((1- lower_threshold)*orig_height),:]
        # Convert temp image to GRAY scale
        img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        height,width=img.shape[:2]

        # Image processing to extract better information form images.
        # Adaptive Biateral Filter:
        img = cv2.adaptiveBilateralFilter(img,ksize=(5,5),sigmaSpace=2)
        # Equalize the histogram to account for better contrast in the images.
        img = cv2.equalizeHist(img);
        # Apply Canny Edge Detector to detect the edges in the image.
        bin_img = cv2.Canny(img,30,60,apertureSize = 3)

        #Thresholds for lane detection. Emperical values, detected from trial and error.
        xl_low = int(-1*orig_width) # low threshold for left x_intercept
        xl_high = int(0.8*orig_width) # high threshold for left x_intercept
        xr_low = int(0.2*orig_width)  # low threshold for right x_intercept
        xr_high = int(2*orig_width) # high threshold for right x_intercept
        xl_phase_threshold = 15  # Minimum angle for left x_intercept
        xr_phase_threshold = 14  # Minimum angle for right x_intercept
        xl_phase_upper_threshold = 80  # Maximum angle for left x_intercept
        xr_phase_upper_threshold = 80  # Maximum angle for right x_intercept

        # Arrays/Containers for intercept values and phase angles.
        xl_arr = np.zeros(xl_high-xl_low)
        xr_arr = np.zeros(xr_high-xr_low)
        xl_phase_arr = []
        xr_phase_arr = []
        # Intercept Bandwidth: Used to assign weights to neighboring pixels.
        intercept_bandwidth = 6

        # Run Probabilistic Hough Transform to extract line segments from Binary image.
        lines=cv2.HoughLinesP(bin_img,rho=1,theta=np.pi/180,threshold=30,minLineLength=20,maxLineGap=5)

        # Loop for every single line detected by Hough Transform
        # print len(lines[0])
        for x1,y1,x2,y2 in lines[0]:
            if(x1<x2 and y1>y2 and x1 < 0.6*width  and x2 > 0.2*width):
                norm = cv2.norm(float(x1-x2),float(y1-y2))
                phase = cv2.phase(np.array(x2-x1,dtype=np.float32),np.array(y1-y2,dtype=np.float32),angleInDegrees=True)
                if(phase<xl_phase_threshold or phase > xl_phase_upper_threshold or x1 > 0.5 * orig_width): #Filter out the noisy lines
                    continue
                xl = int(x2 - (height+lower_threshold*orig_height-y2)/np.tan(phase*np.pi/180))
                # Show the Hough Lines           
                # cv2.line(orig_img,(x1,y1+int(orig_height*upper_threshold)),(x2,y2+int(orig_height*upper_threshold)),(0,0,255),2)

                # If the line segment is a lane, get weights for x-intercepts
                try:
                    for i in range(xl - intercept_bandwidth,xl + intercept_bandwidth):
                        xl_arr[i-xl_low] += (norm**0.5)*y1*(1 - float(abs(i - xl))/(2*intercept_bandwidth))*(phase**2)
                except IndexError:
                    # print "Debug: Left intercept range invalid:", xl
                    continue
                xl_phase_arr.append(phase[0][0])

            elif(x1<x2 and y1<y2 and x2>0.6*width and x1 < 0.8*width):
                norm = cv2.norm(float(x1-x2),float(y1-y2))
                phase = cv2.phase(np.array(x2-x1,dtype=np.float32),np.array(y2-y1,dtype=np.float32),angleInDegrees=True)
                if(phase<xr_phase_threshold or phase > xr_phase_upper_threshold or x2 < 0.5 * orig_width): #Filter out the noisy lines
                    continue
                xr = int(x1 + (height+lower_threshold*orig_height-y1)/np.tan(phase*np.pi/180))
                # Show the Hough Lines           
                # cv2.line(orig_img,(x1,y1+int(orig_height*upper_threshold)),(x2,y2+int(orig_height*upper_threshold)),(0,0,255),2)
                # If the line segment is a lane, get weights for x-intercepts
                try:
                    for i in range(xr - intercept_bandwidth,xr + intercept_bandwidth):
                        xr_arr[i-xr_low] += (norm**0.5)*y2*(1 - float(abs(i - xr))/(2*intercept_bandwidth))*(phase**2)
                except IndexError:
                    # print "Debug: Right intercept range invalid:", xr
                    continue
                xr_phase_arr.append(phase[0][0])
            else:
                pass # Invalid line - Filter out orizontal and other noisy lines.

        # Sort the phase array and get the best estimate for phase angle.
        try:        
            xl_phase_arr.sort()
            xl_phase =  xl_phase_arr[-1] if (xl_phase_arr[-1] < np.mean(xl_phase_arr) + np.std(xl_phase_arr)) else np.mean(xl_phase_arr) + np.std(xl_phase_arr)
        except IndexError:
            # print "Debug: ", fname + " has no left x_intercept information"
            pass
        try:
            xr_phase_arr.sort()
            xr_phase =  xr_phase_arr[-1] if (xr_phase_arr[-1] < np.mean(xr_phase_arr) + np.std(xr_phase_arr)) else np.mean(xr_phase_arr) + np.std(xr_phase_arr)
        except IndexError:
            # print "Debug: ", fname + " has no right x_intercept information"
            pass

        # Get the index of x-intercept (700 is for positive numbers for particle filter.)
        pos_int = np.argmax(xl_arr)+xl_low+700
        # Apply Particle Filter.
        xl_int = xl_int_pf.filterdata(data=pos_int)
        xl_phs = xl_phs_pf.filterdata(data=xl_phase)

        # Draw lines for display
        cv2.line(orig_img,
            (int(xl_int-700), orig_height),
            (int(xl_int-700) + int(orig_height*0.3/np.tan(xl_phs*np.pi/180)),int(0.7*orig_height)),(0,255,255),2)
        # Apply Particle Filter.
        xr_int = xr_int_pf.filterdata(data=np.argmax(xr_arr)+xr_low)
        xr_phs = xr_phs_pf.filterdata(data=xr_phase)
        # Draw lines for display
        cv2.line(orig_img,
            (int(xr_int), orig_height),
            (int(xr_int) - int(orig_height*0.3/np.tan(xr_phs*np.pi/180)),int(0.7*orig_height)),(0,255,255),2)

        # print "Degbug: %5d\t %5d\t %5d\t %5d %s"%(xl_int-700,np.argmax(xl_arr)+xl_low,xr_int,np.argmax(xr_arr)+xr_low,fname)
        intercepts.append((os.path.basename(fname), xl_int[0]-700, xr_int[0]))

        # Show image
        cv2.imshow('Lane Markers', orig_img)
        key = cv2.waitKey(30)
        if key == 27:
            cv2.destroyAllWindows();
            sys.exit(0)

    # CSV output
    with open('intercepts.csv', 'w') as f:
        writer = csv.writer(f)    
        writer.writerows(intercepts)
        
    cv2.destroyAllWindows();

