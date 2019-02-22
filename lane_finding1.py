import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def Image_Process(image):
    low_thresh = 50
    high_thresh = 150
    kernel = 5

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) #converts RGB to grayscale image
    blur = cv.GaussianBlur(gray, (kernel, kernel), 0) #blurs image so that edges are easier to detect
    edges = cv.Canny(blur, low_thresh, high_thresh)

    mask = np.zeros_like(edges)
    ignore_mask_color = 255
    im_copy = image.shape
    #print(im_copy[0]) #y distance
    #print(im_copy[1]) #x distance

    inner_vertice = im_copy[0]
    inner_vertice1 = im_copy[1]

    #data type of vertices must be int32
    vertices = np.array([[(0, inner_vertice), (450, 320), (500, 320), (inner_vertice1, inner_vertice)]], dtype=np.int32)
    cv.fillPoly(mask, vertices, ignore_mask_color)
    masked_edges = cv.bitwise_and(edges, mask)

    #defines what is a "line" in Hough space-->all units in pixels
    rho = 2
    theta = np.pi / 180
    threshold = 15 #min number of intersections in intersections of lines in Hough Space
    min_line_length = 40 #how many pixels to make up a line(think dashed lines)
    max_line_gap = 30   #used for dashed lines and space between them
    line_image = np.copy(image) * 0

    lines = cv.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

    lines_edges = cv.addWeighted(image, 0.8, line_image, 1, 0)

    return cv.imwrite('grey_image2.jpg', lines_edges)


#use to process single frame(in this case jpg)

#path = cv.imread('solidWhiteRight.jpg')
#final_result = Image_Process(path)


#Video Display
video = cv.VideoCapture('solidWhiteRight.mp4')
while video.isOpened():
    try:
        bool, frame = video.read()
        if bool:
            print(frame)
            #passing frame to image processing function
            edited_frame = Image_Process(frame)

#plays AND displays video when you replace "edited_frame" with "frame"-->not displaying video with "edited_frame"
            cv.imshow('Video', edited_frame)
            if cv.waitKey(10) & 0xFF == ord('q'):
                break
        else:
            break
    except:
        pass
video.release()


