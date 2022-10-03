''' This code takes mulitple binary images at multiple thresholds
    and then finds the contours.
    The contour with the highest total area is selected to be drawn.
    Will have to add a min_threshold parameter '''

import cv2 as cv
import numpy as np
import blob_params as bp

# Load source image
im = cv.imread("ex3.tif")
if im is None:
    print('Could not open or find the image:', args.input)
    exit(0)

# Gray and Blur image
im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
#im_gray = cv.equalizeHist(im_gray)
im_blur = cv.GaussianBlur(im_gray, (25,25), 0)

contour_area = []
contour_list = []
for i in range(25,255,10):  # add min threshold for image parameter
    ret, im_thresh = cv.threshold(im_blur, i, 255, cv.THRESH_BINARY)
    # cv.imshow('binary'+str(i), im_thresh)
    # Find only the most external contours
    contours, hierarchy = cv.findContours(im_thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        for j in range(len(contours)):
            contour_list.append(contours[j])  # add contours to main list
            #contour_area.append(cv.contourArea(contours[j]))

# find the contour with the highest area
max_contour = max(contour_list, key=cv.contourArea)
contour_area = cv.contourArea(max_contour)
contour_perimeter = cv.arcLength(max_contour, True)
contour_circularity = bp.get_circularity(max_contour)
cx, cy = bp.get_center(max_contour)

print('Contour Area: ' + str(contour_area))
print('Contour Perimeter: ' + str(round(contour_perimeter,1)))
print('Contour Circularity: ' + str(round(contour_circularity,3)))
print('Contour Center: (' + str(cx) + ',' + str(cy) + ')')

# draw the max contour
im_copy = im.copy()
cv.drawContours(im_copy, max_contour, -1, (0, 255, 0), 2, cv.LINE_8)
cv.circle(im_copy, (cx,cy), 5, (0,0,255), -1)

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


