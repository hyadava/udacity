
**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/calib_image_dist_undist.png "Distorted"
[image2]: ./output_images/road_image_dist_undist.png "Road Transformed"
[image3]: ./output_images/Thresholded_binary.png "Binary Example"
[image4]: ./output_images/perspective_transform.png "Warp Example"
[image5]: ./output_images/polynomial_lane_finding.png "Fit Visual"
[image6]: ./output_images/marked_lanes.png "Marked Lanes"
[image7]: ./output_images/road_img_with_marked_lanes.png "Output"
[video1]: ./proj4_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

###1. Camera Calibration

The code for this step is contained in the code cell# 510 of the IPython notebook located in "./examples/project4.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

###Pipeline (single images)

####2. Undistort a road image.

The code for this step is contained in the code cell # 511 of the IPython notebook located in "./examples/project4.ipynb".

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images. The `mtx` and `dist` objects obtained from the camera caibration step are used to undistort the road image using the `cv2.undistort` function. Since the distortion introduced by the camera remains constant, the calibration matrix can be used to undistort any image taken from the same camera.

![alt text][image2]

####2. Color transforms, gradients or other methods to create a thresholded binary image.

The code for this step is contained in the code cells # 512 and 513 of the IPython notebook located in "./examples/project4.ipynb".

I used a combination of color, absolute x & y, gradient and magnitude thresholds to generate a binary image. The way these thresholds are combined is implemented in the function `get_thresholded_binary_img`. Here's an example of my output for this step.

![alt text][image3]

####3. Perspective transform to create a "bird's eye view" of the road

The code for this step is contained in the code cell # 516 of the IPython notebook located in "./examples/project4.ipynb".

The code for my perspective transform includes a function called `transform_perspective()`. The function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose to hardcode the source and destination points by eyeballing the lanes in the test image straight_lines1.jpg. Since the lane lines are straight in the image, in the bird's eye view they should appear as two roughly parallel lines going from top to the bottom of the image.

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 220, 700      | 220, 700      | 
| 520, 500      | 220, 0        |
| 770, 500      | 1080, 0       |
| 1080, 700     | 1080, 700     |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

####4. Identifying lane-line pixels and fitting their positions with a polynomial.

The code for this step is contained in the code cells 521 and 523 of the IPython notebook located in "./examples/project4.ipynb".

I first identified the starting positions of the lanes at the bottom x-axis of the image by inspecting pixel intensity using a histogram of 'on' pixels in the binary thresholded image. Then using the sliding window technique, described in the lane finding lectures, I traced the lanes going up the image. Finally a second degree polynomial function is used to fit the identified lane points to obtain a smooth lane line. The visualization of the lane lines along with the bounding windows is shown in the image below. The functionality is implemented in the function `find_lanes()` in the code cell 521.

![alt text][image5]

Once the lane lines are identified, a narrow band is generated around the lines to clearly mark the lane lines. The code for this present in the function `mark_known_lanes()` function in the code cell 523.

![alt text][image6]


####5. Calculating the radius of curvature of the lane and the position of the vehicle with respect to center.

The code for this step is contained in the code cells 519 and 523 of the IPython notebook located in "./examples/project4.ipynb".

The curvature of the lane is calculated by evaluating the second order derivative of the polynomial used to fit the lane lines (section 4). The curvature obtained by this derivative is in the pixel space, which can be converted to the radius in the real word by multiplying it by the approximate distance represented by a pixel in the x and y coordinates.

`ym_per_pix = 30/720 # meters per pixel in y dimension`
`xm_per_pix = 3.7/700 # meters per pixel in x dimension`

The code for curvature calculation is in the function `find_curvature()` in the code cell 519.


####6. An example image plotted back down onto the road where the lane area is identified clearly.

The code for this step is contained in the code cells 541 of the IPython notebook located in "./examples/project4.ipynb".


I implemented this step in the function `project_to_road()`. This image is created by un-warping the binary image with the lanes marked and superimposing it on the original un-warped image. The Here is an example of how each step in the pipeline transforms  a test image and the final road image is produced with lanes clearly marked:

![alt text][image7]

---

###Pipeline (video)

####1. Final video output.
For each frame in the video along with the final road image, I also calculate the following

* left and right lane curvatures
* lane width
* left and right lane positions

These values are used to estimate whether the detected lanes make sense. If the detected lane parameters don't fall within the acceptable ranges, the lane parameters detected from earlier frames is used to draw the lane lines. The code which does this estimation is located in the cell 563 of "./examples/project4.ipynb" in the function `lane_finding_pipeline`

Here's a [link to my video result](./proj4_video.mp4)

---

###Discussion

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

####1. Challenges I faced in this project
A lot of the boiler plate code was provided in the lecture exercises and quizes. Even then it was quite challenging to find lanes correctly in the project evaluation video. The biggest problem I faced was the noisy data picked up by the lane detection algorithm using the pixel intensity and sliding window search.

In my implementation I used information from previously detected correct lane positions to smooth over the wrongly detected frames. But that is not very robust as the various parameters in the algorithm are hard coded and will probably work only for roads similar to the one in the video.

####2. Where will your pipeline likely fail  
When the lane background color is not dark enough or when there are light colored dirt markings on the road, the lanes lines are not correctly identified. In my implementation lane positions from earlier frames is used to extrapolate the lanes. This approach will not work if clear lane lines are not found for a large section on the highway, since the saved lane positions will get out of synch with the road.

####3. What could you do to make it more robust
A good way to not depend too much on individual features like clear lane markings and color difference between the road the markings would be to use a CNN based approach to automatically learn more features about the highway. If the algorithm used in this project can be combined with the deep learning approach from project 3 it can create a very robust lane finding solution.
