import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt

test = ('camera_cal/calibration1.jpg')
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
#objp defines number of horizonal white(9) and vertical black(6) intersections(corners) that occur
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('camera_cal/calibration*.jpg')
#print(images)

def UndistortImage(image, objectpoints, imagepoints):
    # im = cv.imread(image)
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objectpoints, imagepoints, image.shape[1:], None, None)
    undistorted_image = cv.undistort(image, mtx, dist, None, mtx)
    return undistorted_image

# Step through the list and search for chessboard corners
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv.findChessboardCorners(gray, (9,6),None)

    # If found, add object points, image points
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        img = cv.drawChessboardCorners(img, (9,6), corners, ret)
        #cv.imshow('img',img)
        #cv.waitKey(500)
        undistort = UndistortImage(img, objpoints, imgpoints)
        cv.imshow("undistorted", undistort)





# cv.waitKey(10000)

#Next step is to find camera calibration matrix and distortion coeffs