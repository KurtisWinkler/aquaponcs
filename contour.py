''' This code takes mulitple binary images at multiple thresholds
    and then finds the contours.
    The contour with the highest total area is selected to be drawn.
    Will have to add a min_threshold parameter '''

import cv2 as cv
import numpy as np

# Load source image
im = cv.imread("ex3.tif")
if im is None:
    print('Could not open or find the image:', args.input)
    exit(0)

# Gray and Blur image
im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
im_blur = cv.GaussianBlur(im_gray, (25,25), 0)

contour_area = []
contour_list = []
for i in range(20,255,10):  # add min threshold for image parameter
    ret, im_thresh = cv.threshold(im_blur, i, 255, cv.THRESH_BINARY)
    # cv.imshow('binary'+str(i), im_thresh)
    # Find only the most external contours
    contours, hierarchy = cv.findContours(im_thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        for j in range(len(contours)):
            contour_list.append(contours[j])  # add contours to main list
            #contour_area.append(cv.contourArea(contours[j]))

# find the contour with the highest area
contour_max_area = max(contour_list, key=cv.contourArea)
#print(contour_area)

# draw the max contour
im_copy = im.copy()
cv.drawContours(im_copy, contour_max_area, -1, (0, 255, 0), 2, cv.LINE_8)

# Display image
cv.imshow('source_window', im)
#cv.imshow('blur', im_blur)
#cv.imshow('binary', im_thresh)
cv.imshow('Contours', im_copy)

cv.waitKey()


'''
# This code is for a single threshold; draws all contours; for reference

# Load source image
im = cv.imread("ex3.tif")
if im is None:
    print('Could not open or find the image:', args.input)
    exit(0)
    
im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
im_blur = cv.GaussianBlur(im_gray, (25,25), 0)

ret, im_thresh = cv.threshold(im_blur, 127, 255, cv.THRESH_BINARY)

# Find contours
contours, hierarchy = cv.findContours(im_thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
print(contours)

# Draw contours
im_copy = im.copy()
cv.drawContours(im_copy, contours, -1, (0, 255, 0), 2, cv.LINE_8)

# Show in a window
cv.imshow('source_window', im)
cv.imshow('blur', im_blur)
cv.imshow('binary', im_thresh)
cv.imshow('Contours', im_copy)

cv.waitKey()
'''


