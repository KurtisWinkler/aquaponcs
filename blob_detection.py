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


def flatten(L, flat_list=None):
    """
    Recursively flattens a nested list into a 1d list
    
    Parameters
    ----------
    L : nested list
        List to be flattened
        
    flat_list : list, optional (default None)
        Flattend list to be returned
        
    Returns
    -------
    flat_list : list
        Flattened version of original list
    """
    # initialize flat list on first call
    if flat_list is None:
        flat_list = []

    for l in L:
        # if inner list, call flatten
        if isinstance(l, list):
            flatten(l, flat_list)
        
        # if not inner list, append value
        else:
            flat_list.append(l)

    return flat_list


def blob_im(im, blobs):
    """
    Draws blob contours on an image
    
    Parameters
    ----------
    im : image matrix
        Image to draw contours on
        
    blobs : list
        List of blobs whos contours will be drawn on image
        
    Returns
    -------
    im_copy: image matrix
        A copy of the original image with contours drawn
    """
    im_copy = im.copy()
    blobs_flat = flatten(blobs)
    contours = [blob.cv_contour for blob in blobs_flat]
    cv.drawContours(im_copy, contours, -1, (0, 255, 0), 2, cv.LINE_8)
    return im_copy


def maxima_filter(contour, local_maxima):
    """
    Returns maxima inside contour if contour contains only 1 maxima
    
    Parameters
    ----------
    contour : list
        Contains contour points
        
    local_maxima : list
        Contains x,y coordinates of local maxima
        
    Returns
    -------
    maxima, idx : tuple of ints, int
        maxima - x,y coordinate of maxima inside contour
        idx - index of maxima in local_maxima list
    """
    points = [cv.pointPolygonTest(np.array(contour), (int(maxima[0]), int(maxima[1])), False) for maxima in local_maxima]

    if points.count(True) == 1:
        idx = points.index(True)
        maxima = local_maxima[idx]
        return maxima, idx

    return None, None


def blob_filter(blob, filters):
    """
    Determines if blob conforms to specified filters
    
    Parameters
    ----------
    blob : Blob object (class Blob)
        Blob to test if it passes filters
        
    filters : nested list
        Each inner list contains:
            1st index - blob parameter
            2nd index - min value of parameter or None
            3rd index - max value of parameter or None
        
    Returns
    -------
    True if blob passes filters
    False otherwise
    
    Example
    -------
    filters = [['area', 20, 500], ['circularity', 0.8, None]]
    
    blob_filter(blob, filters) will return True if the blob has an
    area between 20 and 500 pixels and a circularity greater than 0.8.
    Otherwise it will return False.
    """
    for filt in filters:

        if filt[1]:
            if not (eval('blob.' + filt[0]) >= filt[1]):
                return False

        if filt[2]:
            if not (eval('blob.' + filt[0]) <= filt[2]):
                return False

    return True


def similar_filter(blob_list, params, num=2):
    """
    Returns most similar blobs in a list of blobs
    
    Parameters
    ----------
    blob_list : list of Blob objects (class Blob)
        Contains blobs which will be analyzed for similarity
    
    params : nested list
        Each inner list contains:
            1st index - blob parameter
            2nd index - percent threshold for similarity
            
    num : int, optional (default 2)
        Min value of similar blobs that are needed to both analyze
        similarity and return the similar blobs
        
    Returns
    -------
    sim_blobs - list of Blob objects
        Largest collection of most similar blobs in blob_list
    
    Example
    -------
    params = [['area', 0.2], ['circularity', 0.1]]
    
    blob_filter(blob_list, params) will parse blob_list into lists
    containing blobs that are similar in area (within 20%) and
    circulairty (within 10%). The list with the highest number
    of similar blobs is returned.
    """
    if len(blob_list) >= num:
        sim_list = [[]]
        sim_idx = 0
        last_blob = None
        
        # blobs are sorted by area; largest first
        for blob in blob_list:
            if last_blob is not None:
                filters = []

                for i in range(len(params)):
                    val = eval('last_blob.' + params[i][0])
                    val_range = val * params[i][1]
                    val_min = val - val_range
                    val_max = val + val_range
                    filters.append([params[i][0], val_min, val_max])

                if not blob_filter(blob, filters):
                    sim_list.append([])
                    sim_idx += 1
                    
            sim_list[sim_idx].append(blob)
            last_blob = blob
            
        sim_len = [len(sim) for sim in sim_list]
        sim_max_idx = [idx for idx, item in enumerate(sim_len) if item == max(sim_len)]
        
        sim_blobs = sim_list[sim_max_idx[0]]  # returns largest area blobs if tie
        if len(sim_blobs) >= num:
            return sim_blobs 
        
        else:
            return None
     
    else:
        return None


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


def blob_best(blob_list, criteria):
    '''spits out blob with highest score
       according to criteria, a nested list
       1st index is criteria name
       2nd index is criteria ideal value, 'min', or 'max'
       3rd index is criteria factor'''
    
    if len(blob_list) == 0:
        return None
    
    score = np.zeros(len(blob_list))
    
    for crit in criteria:
        vals = np.array([eval('blob.' + crit[0]) for blob in blob_list])
        
        if crit[1] == 'min':
            rank = [i[0] for i in sorted(enumerate(vals), key=lambda x:x[1], reverse = True)]

        elif crit[1] == 'max':
            rank = [i[0] for i in sorted(enumerate(vals), key=lambda x:x[1])]
            
        elif type(crit[1]) == int or type(crit[1]) == float:
            diff = abs(vals - crit[1])
            rank = [i[0] for i in sorted(enumerate(diff), key=lambda x:x[1], reverse = True)]
        
        for i in range(len(score)):
            if crit[2] == None:
                score[i] += rank.index(i)
            else:
                score[i] += np.multiply(rank.index(i), crit[2])
        
        max_score = max(score)
        max_idx = list(score).index(max_score)
    
    return blob_list[max_idx]


def main():
    im = cv.imread('ex11.tif')
    im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    im_blur = cv.GaussianBlur(im_gray, (15, 15), 0)

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
        cv.circle(im_maxima, (coordinate), 2, (255, 0, 0), 2)

    '''
    filters = [['roughness_perimeter', None, 1.15],
               ['solidity', 0.85, None],
               ['area', None, None],
               ['circularity', 0.4, None]]
    '''
    min_thresh = int(blob_blur.pixel_intensity_median)
    contour_list = [[] for i in range(len(local_max_coords))]
    for i in range(min_thresh, 255, 10):  # add min threshold for image parameter
        _, im_thresh = cv.threshold(im_blur, i, 255, cv.THRESH_BINARY)
        contours, _ = cv.findContours(im_thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        if len(contours) > 0:
            for j in range(len(contours)):
                blob = bc.Blob(contours[j], im)
                key, key_idx = maxima_filter(blob.contour, local_max_coords)
                if key:
                    contour_list[key_idx].append(blob)

    filters = [['area', 25, None], # at least 0.05% of nucleus area
               ['ellipse_fit_mean_residual', None, 2]]

    blob_list = []
    for contours in contour_list:
        blob_list.append([x for x in contours if blob_filter(x, filters)])

    params = [['aspect_ratio', 0.2],
              ['solidity', 0.1],
              ['roughness_perimeter', 0.1],
              ['circularity', 0.2]]

    sim_blobs = [similar_filter(blob, params, 2) for blob in blob_list]
    sim_blobs = [blobs for blobs in sim_blobs if blobs is not None]

    out_filter = ['ellipse_fit_mean_residual']
    no_outs = [outlier_filter(blobs, out_filter) for blobs in sim_blobs]

    criteria = [['area_filled', 'max', None]]
    blobs_best = [blob_best(blobs, criteria) for blobs in no_outs]


    cv.imshow('1. original', im)
    cv.imshow('2. peak local maxima', im_maxima)
    cv.imshow('3. maxima contours', blob_im(im, contour_list))
    cv.imshow('4. filtered contours', blob_im(im, blob_list))
    cv.imshow('5. similar blobs', blob_im(im, sim_blobs))
    cv.imshow('5. no outliers', blob_im(im, no_outs))
    cv.imshow('6. final contours', blob_im(im, blobs_best))
    cv.waitKey()

    
if __name__ == '__main__':
    main()