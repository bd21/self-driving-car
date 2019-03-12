import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt

nx = 9
ny = 6

test = cv.imread('camera_cal/calibration1.jpg')
test_image = cv.imread('test1.jpg')
test3 = cv.imread('test3.jpg')
straight = cv.imread('straight_lines1.jpg')

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
def corners_unwarp(img):

    img_size = (img.shape[1], img.shape[0])

    #Vehicle View
    src = np.float32([[580, 460], [205, 720], [1110, 720], [703, 460]])

    #Desired View
    dst = np.float32([[320, 0], [320, 720], [960, 720], [960, 0]])
    M = cv.getPerspectiveTransform(src, dst)  # The transformation matrix
    Minv = cv.getPerspectiveTransform(dst, src)  # Inverse transformation

    #img = cv.imread('./test_img.jpg')  # Read the test img
    warped_img = cv.warpPerspective(img, M, img_size)  # Image warping
    # Return the resulting image and matrix
    return warped_img


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

#unwarped_image = corners_unwarp(corrected_image, 9, 6)

#color thresh = s_thresh; x gradient = sx_thresh
def pipeline(img, s_thresh=(90, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    # Convert to HLS color space and separate the V channel
    hls = cv.cvtColor(img, cv.COLOR_RGB2HLS)
    h_channel = hls[:, :, 0]
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    # Sobel x
    sobelx = cv.Sobel(l_channel, cv.CV_64F, 1, 0, ksize=1)  # Take the derivative in x
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
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return color_binary, combined_binary


result, result1 = pipeline(corrected_image)

top_down = corners_unwarp(result1)

# plt.imshow(result1)
# plt.imshow(top_down)
# plt.show()


#Sliding Windows Approach
def find_lane_pixels(top_down):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(top_down[top_down.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((top_down, top_down, top_down))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(top_down.shape[0] // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = top_down.nonzero()
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
        win_y_low = top_down.shape[0] - (window + 1) * window_height
        win_y_high = top_down.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv.rectangle(out_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv.rectangle(out_img, (win_xright_low, win_y_low),
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


def fit_polynomial(binary_warped, ym, xm):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    #For Real World Curve Values
    left_fit_c = np.polyfit(lefty*ym, leftx*xm, 2)
    right_fit_c = np.polyfit(righty*ym, rightx*xm, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    try:
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1 * ploty ** 2 + 1 * ploty
        right_fitx = 1 * ploty ** 2 + 1 * ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')

    return out_img, left_fit_c, right_fit_c, ploty



def measure_curvature(left, right, ploty, ym):
    y_eval = np.max(ploty)*ym


    ym_per_pix = 30/720
    xm_per_pix = 3.7/700

    left_curve = ((1 + (2*left[0]*y_eval*ym + left[1])**2)**1.5) / np.absolute(2*left[0])
    right_curve = ((1 + (2*right[0]*y_eval*ym + right[1])**2)**1.5) / np.absolute(2*right[0])


    return left_curve, right_curve

ym = 30/720
xm = 3.7/700


out_img, left_poly, right_ploy, ploty = fit_polynomial(top_down, ym, xm)

left_val, right_val = measure_curvature(left_poly, right_ploy, ploty, ym)

#change if can figure out
#correction = .325

real_left_curvature = left_val
real_right_curvature = right_val


print(real_left_curvature, real_right_curvature)


plt.imshow(out_img)
plt.show()

print('hello')








