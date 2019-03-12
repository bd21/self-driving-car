import time

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt


# The goal of this pipeline is to read in a video, frame by frame,
# and output a video displaying the lane boundaries, lane curvature,
# and vehicle position
#
# @author Blake Denniston (bd21)

# parameters
# calibration
calibration_images = "camera_cal/calibration*.jpg"
nx = 9  # 9 horizontal points on the calibration chessboard
ny = 6  # 6 vertical points


video_location = "Test_Inputs/project_video.mp4"
test_image_location = "TestInputs/straight_lines1.jpg"


# main pipeline controller
def main():
    
    # calibrate camera
    chessboard_images = glob.glob(calibration_images)
    calibrate_camera(chessboard_images)
    
    # load video
    input_video = cv2.imread(video_location)
    # separate into frames

    # for frame in frames:
    #     frame = process_frame()
    #     result.append(frame)
    # 
    # return frame


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

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            plt.imshow(img)
            time.sleep(2)


def process_frame():
    pass


def transform_to_threshold_binary_image():
    pass


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
