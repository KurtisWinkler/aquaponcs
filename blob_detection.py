import blob_class as bc
import math
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from skimage.feature import peak_local_max

def points_in_contour(key_list, contour):
    #keys_found = [key for key in key_list if key in main_list]
    keys_found = []
    keys_idx = []
    for i in range(len(key_list)):
        if cv.pointPolygonTest(np.array(contour), (int(key_list[i][0]),int(key_list[i][1])), False) == True:
            keys_found.append(key_list[i])
            keys_idx.append(i)
    if len(keys_found) == 1:
        return keys_found[0], len(keys_found), keys_idx[0]
    return keys_found, len(keys_found), keys_idx

im = cv.imread('ex111.tif')
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

keys, key_num, key_idx = points_in_contour(local_max_coords, blob_blur.contour)

'''
mask2 = np.zeros(im.shape[0:2])
for i in blob_blur.coords:
    mask2[i[0]][i[1]] = 255
cv.imshow('mask1', blob_blur.image_mask)
cv.imshow('mask2', mask2)
'''

im_copy = im.copy()
for coordinate in local_max_coords:
    cv.circle(im_copy, (coordinate), 2, (0,0,255), 2)

min_thresh = int(blob_blur.pixel_intensity_median)
blob_list = [[] for i in range(len(local_max_coords))]
im_contour_copy = im.copy()
for i in range(min_thresh,255,10):  # add min threshold for image parameter
    _, im_thresh = cv.threshold(im_blur, i, 255, cv.THRESH_BINARY)
    #cv.imshow('binary'+str(i), im_thresh)
    # Find only the most external contours
    contours, _ = cv.findContours(im_thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        for j in range(len(contours)):
            blob = bc.Blob(contours[j], im)
            key, key_num, key_idx = points_in_contour(local_max_coords, blob.contour)
            #cv.imshow('blob ' + str(i) + str(j), blob.image_mask)
            if (blob.roughness_perimeter < 1.1 and  # roughness less than 10%
                blob.solidity > 0.9 and   # solidity less than 10%
                key_num == 1):            # blob only contains 1 maxima
                blob_list[key_idx].append(blob)  # add blobs to main list based on maxima
                #cv.imshow('blob ' + str(i) + str(j), blob.image_masked)
                cv.drawContours(im_contour_copy, blob.contour, -1, (0, 255, 0), 2, cv.LINE_8)
                cv.circle(im_contour_copy, (key), 2, (0,0,255), 2)
                blob_skew_list = [[] for i in range(len(blob_list))]
                min_skew_idx = []
'''
for i in range(len(blob_list)):
    for blob in blob_list[i]:
        blob_skew_list[i].append(abs(blob.pixel_skew))
min_skew_list = [min(blob_skew_list[i]) 
                 if len(blob_skew_list[i]) > 0 else -1 for i in range(len(blob_skew_list))]

min_skew_idx = [blob_skew_list[i].index(min_skew_list[i]) 
                if min_skew_list[i] >= 0 else -1 for i in range(len(blob_skew_list))]

for i in range(len(blob_list)):
    if min_skew_idx[i] != -1:
        blob = blob_list[i][min_skew_idx[i]]
        cv.drawContours(im_contour_copy, blob.contour, -1, (0, 255, 0), 2, cv.LINE_8)
'''

area = [[] for i in range(len(blob_list))]
perimeter = [[] for i in range(len(blob_list))]
circ = [[] for i in range(len(blob_list))]
rough_perim = [[] for i in range(len(blob_list))]
rough_surf = [[] for i in range(len(blob_list))]
skew_list = [[] for i in range(len(blob_list))]
kurt_list = [[] for i in range(len(blob_list))]
for i in range(len(blob_list)):
    for blob in blob_list[i]:
        area[i].append(blob.area)
        perimeter[i].append(blob.perimeter)
        circ[i].append(blob.circularity)
        skew_list[i].append(blob.pixel_skew)
        kurt_list[i].append(blob.pixel_kurtosis)
        rough_perim[i].append(blob.roughness_perimeter)
        rough_surf[i].append(blob.roughness_surface)
    print('area' + str(area[i]))
    print('perimeter' + str(perimeter[i]))
    print('circ' + str(circ[i]))
#    print('skew' + str(skew_list[i]))
#    print('kurt' + str(kurt_list[i]))
#    print('rough_perim' + str(rough_perim[i]))
#    print('rough_surf' + str(rough_surf[i]))

#for blob in blob_list[1]:
#    cv.drawContours(im_contour_copy, blob.contour, -1, (0, 255, 0), 2, cv.LINE_8)
'''      
cv.imshow('original', im)
cv.imshow('peak local maxima', im_copy)
cv.imshow('contours', im_contour_copy)
cv.waitKey()
'''