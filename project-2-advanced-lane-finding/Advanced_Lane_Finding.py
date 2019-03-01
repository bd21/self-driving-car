import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt

nx = 9
ny = 6


test = ('camera_cal/calibration1.jpg')
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
#objp defines number of horizonal white(9) and vertical black(6) intersections(corners) that occur
objp = np.zeros((ny*nx,3), np.float32)
objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('camera_cal/calibration*.jpg')
#print(images)

# make the lines straight
def undistort_image(image, objectpoints, imagepoints):
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objectpoints, imagepoints, image.shape[1:], None, None)
    undistorted_image = cv.undistort(image, mtx, dist, None, mtx)
    return undistorted_image, mtx, dist


# fit the image according to the source points
def warp_image(img, nx, ny, mtx, dist):

    # convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (nx, ny), None)

    if ret == True:
        # If we found corners, draw them! (just for fun)
        cv.drawChessboardCorners(img, (nx, ny), corners, ret)

        # Choose offset from image corners to plot detected corners
        # This should be chosen to present the result at the proper aspect ratio
        # offset
        offset = 100
        # Grab the image shape
        img_size = (gray.shape[1], gray.shape[0])

        # For source points I'm grabbing the outer four detected corners
        src = np.float32([corners[0], corners[nx - 1], corners[-1], corners[-nx]])
        # For destination points, I'm arbitrarily choosing some points to be
        # a nice fit for displaying our warped result
        # again, not exact, but close enough for our purposes
        dst = np.float32([[offset, offset], [img_size[0] - offset, offset],
                          [img_size[0] - offset, img_size[1] - offset],
                          [offset, img_size[1] - offset]])
        # Given src and dst points, calculate the perspective transform matrix
        M = cv.getPerspectiveTransform(src, dst)
        # Warp the image using OpenCV warpPerspective()
        warped = cv.warpPerspective(img, M, img_size)


        # Return the resulting image and matrix
    return warped


# Step through the list and search for chessboard corners
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv.findChessboardCorners(gray, (nx, ny),None)

    # If found, add object points, image points
    if ret:
        print("found chessboard corners")
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        # img = cv.drawChessboardCorners(img, (nx, ny), corners, ret)
        print("showing distorted image")
        cv.imshow('img', img)

        print("showing corrected image")
        corrected_image, mtx, dist = undistort_image(img, objpoints, imgpoints)
        cv.imshow('img2', corrected_image)


        print("showing unwarped image")
        cv.imshow('img3', warp_image(corrected_image, nx, ny, mtx, dist))
        cv.waitKey(4000) # wait 4 seconds



# uncomment these next 4 lines for the test image
# undistort = UndistortImage(test, objpoints, imgpoints)
# #cv.imshow("blah", cv.imread(test))
# cv.imshow("undistorted", undistort)
# cv.waitKey(10000)

#Next step is to find camera calibration matrix and distortion coeffs