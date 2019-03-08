import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt

nx = 9
ny = 6


test = cv.imread('camera_cal/calibration1.jpg')
test_image = cv.imread('test1.jpg')
#test1 = cv.imread('Test_inputs/test1.jpg')

#prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
#objp defines number of horizonal white(9) and vertical black(6) intersections(corners) that occur
objp = np.zeros((ny*nx,3), np.float32)
objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
chessboard_images = glob.glob('camera_cal/calibration*.jpg')

# make the lines straight
def undistort_image(image, objectpoints, imagepoints):
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objectpoints, imagepoints, image.shape[1:], None, None)
    undistorted_image = cv.undistort(image, mtx, dist, None, mtx)
    return undistorted_image, mtx, dist


# fit the image according to the source points
def Warp_Image(img, Nx, Ny):
    # convert to grayscale
    gray1 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    #############WHY NOT FINDING CORNERS??????????????????????????????
    ret1, corners = cv.findChessboardCorners(gray1, (Nx, Ny), None)

    if ret1 == True:
        # If we found corners, draw them! (just for fun)
        cv.drawChessboardCorners(img, (Nx, Ny), corners, ret)

        # Choose offset from image corners to plot detected corners
        # This should be chosen to present the result at the proper aspect ratio
        # offset
        offset = 100
        # Grab the image shape
        img_size = (gray.shape[1], gray.shape[0])

        # For source points I'm grabbing the outer four detected corners
        src = np.float32([corners[0], corners[Nx - 1], corners[-1], corners[-Nx]])
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
    else:
        warped = "did not find any corners"

        # Return the resulting image and matrix
    return warped



# Step through the list and search for chessboard corners
#purpose below is to create/fill objpoints, imgpoints lists
for fname in chessboard_images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv.findChessboardCorners(gray, (nx, ny), None)

    # If found, add object points, image points
    if ret:
       # print("found chessboard corners")
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        img = cv.drawChessboardCorners(img, (nx, ny), corners, ret)


#dont need to return dist, mtx?
corrected_image, mtx, dist = undistort_image(test_image, objpoints, imgpoints)
cv.imshow('test', corrected_image)

def abs_sobel_binary(img, thresh_min=0, thresh_max=255, kernel_size=3):
    #change color transformation to something else
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow('test', gray)
    #cv.sobel calculates gradient(derivative) in x, y direction
    sobel_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=kernel_size)
    sobel_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=kernel_size)
    gradient_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    #rescale to 8bit
    scale = np.max(gradient_mag)/255
    gradient_mag = (gradient_mag/scale).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradient_mag)
    #inclusive threshhold
    binary_output[(gradient_mag >= thresh_min) & (gradient_mag <= thresh_max)] = 1
    cv.imshow('test', binary_output)
    return binary_output

hls = cv.cvtColor(img, cv.COLOR_RGB2HLS)
h_channel = hls[:,:,0]
l_channel = hls[:,:,1]
s_channel = hls[:,:,2]

def color_binary(img, channel = s_channel, color_min = 170, color_max = 255):
    # hls replaces bgr in normal picture data type with Hue, Lightness, Value
    # eg if you wanted to make all h pixels zero --> h_channel = 0
    # see link: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_core/py_basic_ops/py_basic_ops.html
    s_binary = np.zeros_like(s_channel)
    s_binary[(channel >= color_min) & (channel <= color_max)] = 1
    return s_binary



sobel_binary = abs_sobel_binary(corrected_image)
col_binary = color_binary(corrected_image)
cv.imshow('sobel', sobel_binary)
cv.imshow('col', col_binary)

combined_binary = np.zeros_like(sobel_binary)
combined_binary[(col_binary == 1) | (sobel_binary == 1)] = 1

cv.imshow('combined',combined_binary)
cv.waitKey(4000)






