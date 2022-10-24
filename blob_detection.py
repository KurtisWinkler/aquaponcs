'''Strategy for finding blobs:
    1. Find maxima in image
    2. Find contours that only contain a single maxima
    3. Keep contours that also have specific ellispe fit residuals and area
    4. Remove contours that are outliers when looking at ellipse fit residuals
    5. Choose the blob with the largest area as the contour (will update)
'''

import blob_class as bc
import math
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from skimage.feature import peak_local_max
from scipy.stats import zscore


def flatten(L , flat_list = None):

    if flat_list is None:
        flat_list = []

    for l in L:
        if isinstance(l, list):
            flatten(l, flat_list)
        else:
            flat_list.append(l)
            
    return flat_list


def blob_im(im, L):
    im_copy = im.copy()
    L_flat = flatten(L)
    contours = [blob.cv_contour for blob in L_flat]
    cv.drawContours(im_copy, contours, -1, (0, 255, 0), 2, cv.LINE_8)
    return im_copy


def maxima_filter(contour, local_maxima):
    ''' returns maxima inside the contour if the contour
        only contains 1 maxima'''
    points = [cv.pointPolygonTest(np.array(contour), (int(maxima[0]), int(maxima[1])), False) for maxima in local_maxima]

    if points.count(True) == 1:
        idx = points.index(True)
        maxima = local_maxima[idx]
        return maxima, idx

    return None, None


def blob_filter(blob, filters):
    ''' filters is a nested list: each inner list contains the 
        parameter as first index, min value as 2nd index, and max value as 3rd index '''
    for filt in filters:

        if filt[1]:
            if not (eval('blob.' + filt[0]) >= filt[1]):
                return False

        if filt[2]:
            if not (eval('blob.' + filt[0]) <= filt[2]):
                return False

    return True


def outlier_filter(blob_list, params):
    ''' params is a list to filter the blobs by, the blobs 
        will be filtered according to the order in the list'''

    blob_copy = np.array(blob_list.copy())
    
    for param in params:
        vals = [eval('blob.' + param) for blob in blob_copy]
        zscores = np.array(zscore(vals))
        mean = np.mean(vals)
        safe = [(0.1 * mean) - mean, (0.1 * mean) + mean]   # 10% above/below mean is fine
        blob_copy = blob_copy[np.where(((zscores >= -1) & (zscores <= 1))
                                     | ((vals >= safe[0]) & (vals <= safe[1])))]
        
    return list(blob_copy)


def blob_best(blob_list):
    '''spits out blob with highest score'''
    if len(blob_list) == 0:
        return None

    max_area = max([blob.area for blob in blob_list])
    score_area = [blob.area / max_area for blob in blob_list]

    scores = [sum(i) for i in zip(score_area)] #, score_circularity, score_roughness_perimeter, score_solidity)]

    if scores:
        max_value = max(scores)
        max_index = scores.index(max_value)
    
    return blob_list[max_index]


im = cv.imread('ex3.tif')
im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
im_blur = cv.GaussianBlur(im_gray,(15,15),0)

'''example mask for finding nuclues -> now to find blobs'''
ret, im_binary = cv.threshold(im_blur, 25, 255, cv.THRESH_BINARY)
contours, hierarchy = cv.findContours(im_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
contour = max(contours, key=cv.contourArea)
blob_blur = bc.Blob(contour, im_blur)
#im_mask_blur = cv.GaussianBlur(blob.image_masked, (15,15),0)

local_max_thresh = blob_blur.pixel_intensity_percentile(80)
local_max_coords = peak_local_max(blob_blur.image_masked, min_distance=20, threshold_abs=local_max_thresh)
local_max_coords = [[x, y] for y, x in local_max_coords]  # switch to x,y

im_maxima = im.copy()
for coordinate in local_max_coords:
    cv.circle(im_maxima, (coordinate), 2, (255,0,0), 2)

'''
filters = [['roughness_perimeter', None, 1.15],
           ['solidity', 0.85, None],
           ['area', None, None],
           ['circularity', 0.4, None]]
'''
min_thresh = int(blob_blur.pixel_intensity_median)
contour_list = [[] for i in range(len(local_max_coords))]
for i in range(min_thresh,255,10):  # add min threshold for image parameter
    _, im_thresh = cv.threshold(im_blur, i, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(im_thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        for j in range(len(contours)):
            blob = bc.Blob(contours[j], im)
            key, key_idx = maxima_filter(blob.contour, local_max_coords)
            if key:
                contour_list[key_idx].append(blob)
                
filters = [['area', 25, None], # at least 0.05% of nucleus area
           ['ellipse_fit_mean_residual', None, 1]]      

blob_list = []
for contours in contour_list:
    blob_list.append([x for x in contours if blob_filter(x, filters)])

blob_list = [x for x in blob_list if len(x) >= 2] # remove if not enough blobs for maxima
out_filter = ['ellipse_fit_mean_residual']

no_outs = [outlier_filter(blobs, out_filter) for blobs in blob_list] 
                
blobs_best = [blob_best(blobs) for blobs in no_outs]


cv.imshow('1. original', im)
cv.imshow('2. peak local maxima', im_maxima)
cv.imshow('3. maxima contours', blob_im(im, contour_list))
cv.imshow('4. filtered contours', blob_im(im, blob_list))
cv.imshow('5. no outliers', blob_im(im, no_outs))
cv.imshow('6. final contours', blob_im(im, blobs_best))
cv.waitKey()
