## Lane Detection and Tracking

Sample code to detect lanes in the highway images and track it's x-intercept value in the image.

### Image Processing

To account for real world lighting conditions, we can use **_Adaptive Bilateral Filtering_** (selectively smooth the image and preserves the edges) and **_Histogram Equalizations_**. Below pictures show the results.

###### Adaptive Bilateral Filtering
![bilateral_filter][smoothing]

###### Histogram Equalization
![hist_1][hist_1]
![hist_2][hist_2]


### Lane Detection

From the enhanced images, **_Canny Edge Detector_** is used to detect the edges in the image and filter out the noisy edges. Using this binay image from the edge detctor, **_Hough Transform_** is used to extract the line information. All the detected lines are used to vote for the possible x-intercept value and strength proportional to their importance and location. Below is one example:

###### Canny Edge Detection
![canny][canny]

###### Lane Detection
![lane][lane]

### Tracking
Once the lane is detected in one image, it's x-intercept values is used in **_Particle Filter_** to track the x-intercept values in the next images from the video sequence.

### Demo Video:
[[Watch on YouTube](http://www.youtube.com/watch?v=xsndYoFf7Pw)]

[smoothing]:readme_images/smoothing.png
[hist_1]:readme_images/hist_1.png
[hist_2]:readme_images/hist_2.png
[canny]:readme_images/canny.png
[lane]:readme_images/lane.png
