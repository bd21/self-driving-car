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

# parameters
calibration_images = "camera_cal/calibration*.jpg"
nx = 9  # 9 horizontal points on the calibration chessboard
ny = 6  # 6 vertical points


video_location = "Test_Inputs/project_video.mp4"
test_image_location = "Test_Inputs/straight_lines1.jpg"


# main pipeline controller
def main():
    # calibrate camera
    # chessboard_images = glob.glob(calibration_images)
    # mtx, dist = calibrate_camera(chessboard_images)

    # load video
    # input_video = cv2.imread(video_location)
    # separate into frames

    img = mpimg.imread(test_image_location)
    new_img = process_frame(img)
    cv2.imshow('img', new_img)
    cv2.waitKey(6000)

    # for frame in frames:
    #     frame = process_frame()
    #     result.append(frame)
    #
    # return frame

def process_frame(img):
    # create thresholded binary image
    tbi_image = transform_to_threshold_binary_image(img)
    return tbi_image


def calibrate_camera(chessboard_images):

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

        # get the camera matrix, distortion coefficients, rotation and translation vectors
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        return mtx, dist


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


def transform_to_birds_eye_view():
    pass


def detect_and_fit_lane_lines():
    pass


def calculate_curvature():
    pass


def warp_boundaries_onto_frame():
    pass


def get_vehicle_position():
    pass


# so we can use functions before their declaration
if __name__ == '__main__':
    main()
