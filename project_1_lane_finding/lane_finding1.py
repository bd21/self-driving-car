import cv2 as cv
import numpy as np


# process a single frame of an image
def process_image(image):
    low_thresh = 50
    high_thresh = 150
    kernel = 5

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # converts RGB to grayscale image
    blur = cv.GaussianBlur(gray, (kernel, kernel), 0)  # blurs image so that edges are easier to detect
    edges = cv.Canny(blur, low_thresh, high_thresh)

    mask = np.zeros_like(edges)
    ignore_mask_color = 255
    im_copy = image.shape

    inner_vertice = im_copy[0]
    inner_vertice1 = im_copy[1]

    # data type of vertices must be int32
    vertices = np.array([[(0, inner_vertice), (450, 320), (500, 320), (inner_vertice1, inner_vertice)]], dtype=np.int32)
    cv.fillPoly(mask, vertices, ignore_mask_color)
    masked_edges = cv.bitwise_and(edges, mask)

    # defines what is a "line" in Hough space-->all units in pixels
    rho = 2
    theta = np.pi / 180
    threshold = 20  # min number of intersections in intersections of lines in Hough Space
    min_line_length = 40  # how many pixels to make up a line(think dashed lines)
    max_line_gap = 120   # used for dashed lines and space between them
    line_image = np.copy(image) * 0

    lines = cv.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

    lines_edges = cv.addWeighted(image, 0.8, line_image, 1, 0)

    return lines_edges


# save the video in the root directory as a avi
def save_video():
    video = cv.VideoCapture('Example_Sources/solidWhiteRight.mp4')

    frame_width = int(video.get(3))
    frame_height = int(video.get(4))

    # define codec, create VideoWriter
    out = cv.VideoWriter('output.avi', cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

    # iterate through video and process frame by frame
    while video.isOpened():
        bool, frame = video.read()

        if bool:
            edited_frame = process_image(frame)
            out.write(edited_frame)

        else:
            break

    # Release everything if job is finished
    video.release()
    out.release()
    cv.destroyAllWindows()


# run
save_video()
