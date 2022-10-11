import blob_class as bc
import math
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from skimage.feature import peak_local_max

im = cv.imread('ex3.tif')
im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
im_blur = cv.GaussianBlur(im_gray,(15,15),0)

'''example mask for finding nuclues -> now to find blobs'''
ret, im_binary = cv.threshold(im_blur, 25, 255, cv.THRESH_BINARY)
contours, hierarchy = cv.findContours(im_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
contour = max(contours, key=cv.contourArea)
blob_blur = bc.Blob(im_blur, contour)
#im_mask_blur = cv.GaussianBlur(blob.image_masked, (15,15),0)

local_max_thresh = blob_blur.pixel_intensity_percentile(80)
local_max_coords = peak_local_max(blob_blur.image_masked, min_distance=20, threshold_abs=local_max_thresh)

im_copy = im.copy()
for coordinate in local_max_coords:
    cv.circle(im_copy, (coordinate[1],coordinate[0]), 2, (0,255,0), 2)

min_thresh = int(blob_blur.pixel_intensity_median)
blob_list = []
im_contour_copy = im.copy()
for i in range(min_thresh,255,10):  # add min threshold for image parameter
    _, im_thresh = cv.threshold(im_blur, i, 255, cv.THRESH_BINARY)
    #cv.imshow('binary'+str(i), im_thresh)
    # Find only the most external contours
    contours, _ = cv.findContours(im_thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        for j in range(len(contours)):
            blob = bc.Blob(im, contours[j])
            if (blob.roughness < 1.1 and
                blob.solidity > 0.9):
                blob_list.append(blob)  # add blobs to main list
                #cv.imshow('blob ' + str(i) + str(j), blob.image_masked)
                cv.drawContours(im_contour_copy, blob.contour, -1, (0, 255, 0), 2, cv.LINE_8)
            
            
'''
cv.imshow('peak local maxima', im_copy)
cv.imshow('blob with mask', blob_blur.image_masked)
'''
cv.imshow('contours', im_contour_copy)
cv.waitKey()
