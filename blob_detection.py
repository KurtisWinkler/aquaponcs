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


def contour_maxima(contour, local_maxima):
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
        
    return blob_copy


def blob_scores(blob_list):
    '''spits out a score for each blob in a list'''
    if len(blob_list) == 0:
        return None
    max_area = max([blob.area for blob in blob_list])
    max_circularity = max([blob.circularity for blob in blob_list])
    # make max value of roughness_perimeter 1 by taking inverse
    max_roughness_perimeter = max([1/blob.roughness_perimeter for blob in blob_list])
    max_solidity = max([blob.solidity for blob in blob_list])

    score_area = [blob.area / max_area for blob in blob_list]
    score_circularity = [blob.circularity / max_circularity for blob in blob_list]
    score_roughness_perimeter = [(1/blob.roughness_perimeter) / max_roughness_perimeter for blob in blob_list]
    score_solidity = [blob.solidity / max_solidity for blob in blob_list]
    '''
    residuals = [blob.ellipse_fit_mean_residual for blob in blob_list]
    aspect_ratio = [blob.aspect_ratio for blob in blob_list]
    roundness = [blob.roundness for blob in blob_list]
    circularity = [blob.circularity for blob in blob_list]
    resid_corr = [blob.ellipse_fit_mean_residual/blob.perimeter for blob in blob_list] #SEEMS TO WORK GREAT
    zscores = zscore(resid_corr)
    print('zscores: ' + str(zscores))
    print('residuals: ' + str(residuals))
    print('resid_corr: ' + str(resid_corr))

    print('aspect_ratio: ' + str(aspect_ratio))
    print('roundness: ' + str(roundness))
    print('circularity: ' + str(circularity))
    '''
    score_circularity = np.multiply(score_circularity, 0)
    score_roughness_perimeter = np.multiply(score_roughness_perimeter, 0)
    score_solidity = np.multiply(score_solidity, 0)

    scores = [sum(i) for i in zip(score_area, score_circularity, score_roughness_perimeter, score_solidity)]
    '''
    print('area' + str(score_area))
    print('circularity' + str(score_circularity))
    print('rough' + str(score_roughness_perimeter))
    print('solid' + str(score_solidity))
    print('Final scores' + str(scores))
    '''
    return scores

im = cv.imread('ex11.tif')
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

im_copy = im.copy()
for coordinate in local_max_coords:
    cv.circle(im_copy, (coordinate), 2, (0,0,255), 2)
    
filters = [['area', 25, None], # at least 0.05% of nucleus area
           ['ellipse_fit_mean_residual', None, 1]]
'''
filters = [['roughness_perimeter', None, 1.15],
           ['solidity', 0.85, None],
           ['area', None, None],
           ['circularity', 0.4, None]]
'''
min_thresh = int(blob_blur.pixel_intensity_median)
blob_list = [[] for i in range(len(local_max_coords))]
im_contours_copy = im.copy()
im_contour_copy = im.copy()
im_outlier_copy = im.copy()
for i in range(min_thresh,255,10):  # add min threshold for image parameter
    _, im_thresh = cv.threshold(im_blur, i, 255, cv.THRESH_BINARY)
    #cv.imshow('binary'+str(i), im_thresh)
    # Find only the most external contours
    contours, _ = cv.findContours(im_thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        for j in range(len(contours)):
            blob = bc.Blob(contours[j], im)
            key, key_idx = contour_maxima(blob.contour, local_max_coords)
            #cv.imshow('blob ' + str(i) + str(j), blob.image_mask)
            if key and blob_filter(blob,filters):            # blob only contains 1 maxima
                blob_list[key_idx].append(blob)  # add blobs to main list based on maxima
                #cv.imshow('blob ' + str(i) + str(j), blob.image_masked)
                cv.drawContours(im_contours_copy, blob.cv_contour, -1, (0, 255, 0), 2, cv.LINE_8)
                #cv.circle(im_contour_copy, (key), 2, (0,0,255), 2)

no_outliers = []
for blob_inner_list in blob_list:
    if len(blob_inner_list) >= 2:
        no_outs = outlier_filter(blob_inner_list, ['ellipse_fit_mean_residual'])
                                                  
        no_outliers.append(no_outs)
        for blob in no_outs:
            cv.drawContours(im_outlier_copy, blob.cv_contour, -1, (0, 255, 0), 2, cv.LINE_8)
                
for blob_inner_list in no_outliers:
    scores = blob_scores(blob_inner_list)
    if scores:
        max_value = max(scores)
        max_index = scores.index(max_value)
        cv.drawContours(im_contour_copy, blob_inner_list[max_index].cv_contour, -1, (0, 255, 0), 2, cv.LINE_8)


#for blob in blob_list[1]:
#    cv.drawContours(im_contour_copy, blob.cv_contour, -1, (0, 255, 0), 2, cv.LINE_8)

cv.imshow('original', im)
#cv.imshow('peak local maxima', im_copy)
cv.imshow('all_contours', im_contours_copy)
cv.imshow('contours', im_contour_copy)
cv.imshow('no outliers', im_outlier_copy)
cv.waitKey()
