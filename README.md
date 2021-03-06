# EC601-Fall2017-Seamless-Track-Detection
Boston University College of Engineering - Fall 2017

## Trek - Lane Detection Project

**TREK** (Tracking Rural Environments Keenly) is a semester-long project for Boston University's EC601 Product Design course. We aim to build a lane detection algorithm for unmarked roads. Our program is based on other files found on Github (links below). 

**Tested on**: Python 3.5, OpenCV 3.1.0, NumPy 1.13.3

### Using Our Program

Our program is based on the **George Sung** lane detection program, and runnable files of the original are located in that directory. The main program to run is line_fit_video.py located in the /our_main_code/ folder which will use an input video, detect/fit lanes, and produce an output video. Be sure to have the above listed versions of Python, OpenCV, and NumPy installed.

## Images and Videos

Images and videos are from the [KITTI Vision Benchmark Suite](http://www.cvlibs.net/datasets/kitti/index.php) used under the [Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License](https://creativecommons.org/licenses/by-nc-sa/3.0/) for academic purposes only.

## Lane Detection Programs Tested

[**Advanced Lane Detection by George Sung**](https://github.com/georgesung/advanced_lane_detection) - Working on Python3.5,OpenCV 3.1.0, NumPy 1.13.3. Requires particular files/information for camera calibration.

[**Lane Tracking - by FrenkT**](https://github.com/FrenkT/LaneTracking) - Fails to run on Python3.5,OpenCV 3.1.0, NumPy 1.13.3, but upon further inspection it does not actually do anything. Originally added it due to the ReadMe saying it used Kalman filtering for lane tracking. 

[**Lane Detection - by vinaykumarhs2020**](https://github.com/vinaykumarhs2020/lane_detection) - Program is only for detecting lanes on a raod in picture form and the code that is demoed for video is not included. 

[**Machine Learning MLND-Capstone - by mvirgo**](https://github.com/mvirgo/MLND-Capstone) - 

[**Advanced-Lane-and-Vehicle-Detection - by muddassir235**](https://github.com/muddassir235/Advanced-Lane-and-Vehicle-Detection) -  Program does more than we needed it to and we were unable to get the program file to run.  Large program only linked insead of reuploading.

[**Advanced-Lane-Lines - by NikolasEnt**](https://github.com/NikolasEnt/Advanced-Lane-Lines) - First program we tried. 

## References

1. Muad *et al*. **Implementation of Inverse Perspective Maping Algorithm for the Development of an Automatic Lane Tracking System**. IEEE, May 2005. [link](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=1414393&tag=1)

2. Wan-zhi and Zeng-cai. **Rural Road Detection of Color Image in Complicated Environment**. International Journal of Signal Processing, Image Processing and Pattern Recognition, Vol.6, No.6(2013), pp.161-168 [link](http://www.sersc.org/journals/IJSIP/vol6_no6/15.pdf)

3. Seo, Young-Woo. **Detection and Tracking of Boundary of Unmarked Roads**. IEEE. October 2014 [link](http://ieeexplore.ieee.org/document/6916256/?arnumber=6916256)
