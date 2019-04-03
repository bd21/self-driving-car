import time

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# The goal of this pipeline is to read in a video, frame by frame,
# and output a video displaying the lane boundaries, lane curvature,
# and vehicle position
#
# @author Blake Denniston (bd21)


# calibration parameters
calibration_images = "camera_cal/calibration*.jpg"
nx = 9  # 9 horizontal points on the calibration chessboard
ny = 6  # 6 vertical points


# input parameters
video_location = "Test_Inputs/project_video.mp4"
test_image_location = "Test_Inputs/test3.jpg"  # todo delete


# pipeline parameters




# main pipeline controller
def main():

    # calibrate camera todo uncomment
    # chessboard_images = glob.glob(calibration_images)
    # ret, mtx, dist, rvecs, tvecs = calibrate_camera(chessboard_images)

    # load video
    input_video = cv2.imread(video_location)
    # separate into frames

    start = time.time()

    img = mpimg.imread(test_image_location)
    new_img = process_frame(img, start)
    cv2.imshow('img', new_img)
    cv2.waitKey(7000)
    #
    # for frame in frames:
    #     frame = process_frame()
    #     result.append(frame)

    # return frame

def process_frame(img):

    img_size = (img.shape[1], img.shape[0])

    temp_img = img.copy()

    # apply distortion correction to image
    # img = undistort_frame(img)

    # create thresholded binary image
    temp_img = transform_to_threshold_binary_image(temp_img)

    # transform to birds eye view
    temp_img, M, Minv = undistort_frame(temp_img, img_size)

    # detect lane pixels and extract the left and right lane lines
    left_fit, right_fit = get_lane_lines(temp_img)

    # measure curvature todo

    # draw these lines on a blank screen
    lane_lines_img = draw_lane_lines(img_size, left_fit, right_fit)

    # transform the fitted lines from the birds eye view to the original view
    lane_lines_img = cv2.warpPerspective(lane_lines_img, Minv, img_size)

    # combine this with the original image
    temp_img = cv2.addWeighted(img, 1, lane_lines_img, 0.8, 0)

    return temp_img

# helper functions


def draw_lane_lines(img_size, left_fit, right_fit):

    height = img_size[1]
    width = img_size[0]
    color_warp = np.zeros((height, width, 3), np.uint8)

    ploty = np.linspace(0, width, height)

    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    return color_warp


# calibrate the camera based on:
#   1. a set of input images
#   2. the number of horizontal intersections
#   3. the number of vertical intersections
def calibrate_camera(chessboard_images, nx, ny):

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    # objp defines number of horizontal white(9) and vertical black(6) intersections(corners) that occur
    objp = np.zeros((ny * nx, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    for fname in chessboard_images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, add object points, image points
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Uncomment to draw and display the corners frame by frame
            # img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            # cv2.imshow('img', img)
            # cv2.waitKey(2000)

        # return the camera matrix, distortion coefficients, rotation and translation vectors
        return cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


# undistort frame manually
def undistort_frame(img, img_size):

    # Vehicle View
    src = np.float32([[580, 460], [205, 720], [1110, 720], [703, 460]])

    # Desired View
    dst = np.float32([[320, 0], [320, 720], [960, 720], [960, 0]])
    M = cv2.getPerspectiveTransform(src, dst)  # The transformation matrix
    Minv = cv2.getPerspectiveTransform(dst, src) # Inverse transformation matrix

    warped_img = cv2.warpPerspective(img, M, img_size)  # Image warping

    # Return the resulting image and matrix
    return warped_img, M, Minv

def transform_to_threshold_binary_image(img):

    s_thresh = (170, 255)
    sx_thresh = (20, 100)

    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

    # convert back to greyscale
    color_binary = cv2.cvtColor(color_binary, cv2.COLOR_BGR2GRAY)

    return color_binary

# this function is taking the longest time
def get_lane_lines(img):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = detect_and_fit_lane_lines(img)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')

    return left_fit, right_fit

# detect lane lines with the "sliding windows" approach
def detect_and_fit_lane_lines(img):

    # Take a histogram of the bottom half of the image
    histogram = np.sum(img[img.shape[0] // 2:, :], axis=0)

    # Create an output image to draw on and visualize the result
    out_img = np.dstack((img, img, img)) * 255

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 10
    # Set the width of the windows +/- margin
    margin = 180
    # Set minimum number of pixels found to recenter window
    minpix = 40

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(img.shape[0] // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window + 1) * window_height
        win_y_high = img.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def calculate_curvature():
    pass


def warp_boundaries_onto_frame():
    pass


def get_vehicle_position():
    pass


# so we can use functions before their declaration
# if __name__ == '__main__':
#     main()
