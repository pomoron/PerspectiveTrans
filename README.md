# PerspectiveTrans
Several handcrafted methods to convert images taken from a perspective into a bird's eye view

# Introductions

All methods essentially attempted to recover either the transformation matrix or the transformed four corners between the original and warped image.
The transformed coordinates of the four corners are particularly helpful as the warping module in OpenCV (cv2.getPerspectiveTransform) takes in old and new coordinates as the two arguments.

## Method 1 - By homography
Not yet completed as we don't have the intrinsic parameters of the camera when the Jupyter notebook was written.

## Method 2 - Geometric Transform
Reference in https://stackoverflow.com/questions/17087446/how-to-calculate-perspective-transform-for-opencv-from-rotation-angles
Module "perspective" cloned from https://github.com/manisoftwartist/perspectiveproj.git 

## Method 3 - Geometric Transform (2)
To estimate positions of the four corners of an image in the real world. Essentially by reversing the pitch, roll and yaw when the image was taken.
Idea follow Bertozzi et al. (1997) Stereo inverse perspective mapping: theory and applications
Implementation inspired by Kong et al. (2022) Automatic Detection and Assessment of Pavement Marking Defects with Street View Imagery at the City Scale

## Method 4 - Hard-coded solutions
Image warped in photo editing software and the coordinates of the four corners recorded.
Coordinates of the original and the warped image are hard-coded in the codes for transformation.

# Implementation
1. Git clone the repo to your local directory

2. File placement

  Your original images should go into the folder "Input".

  The reference file should go into the folder "Reference". This is the file that the jupyter notebook reads in the pitch, roll and yaw angles.

3. The Jupyter Notebook
The 4 methods are documented in the jupyter notebook.
Some methods require import of perspective.py cloned from the aforementioned git. Please do not delete the file from the directory.

Enjoy!
