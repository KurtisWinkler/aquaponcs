"""
blob_detection.py contains functions for detecting blobs,
drawing blob contours, and filtering for the ideal blobs
Functions
---------
flatten - Recursively flattens a nested list into a 1d list
blob_im - Draws blob contours on an image
get_maxima - Returns local maxima of blob/image
get_contours - Returns contours of image
segment_contours - segments contours based on maxima
organize_contours - organizes contours based on maxima
maxima_filter - Returns maxima if contour contains 1 maxima
blob_filter - Returns True if blob conforms to specified filters
similar_filter - Returns most similar blobs in a list of blobs
outlier_filter - Removes outliers from list of blobs
blob_best - Returns blob with highest score out of a list of blobs
final_blobs_filter - Removes overlapping, unnessessary blobs
"""

import blob_class as bc
import math
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from skimage.feature import peak_local_max
from scipy.stats import zscore
from skimage.segmentation import watershed
from scipy import ndimage as ndi


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
    if not isinstance(L, (list, np.ndarray)):
        raise TypeError('L must be a list or numpy array')

    # initialize flat list on first call
    if flat_list is None:
        flat_list = []

    for val in L:
        # if inner list, call flatten
        if isinstance(val, list):
            flatten(val, flat_list)

        # if not inner list, append value
        else:
            flat_list.append(val)

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
    if not isinstance(im, (list, np.ndarray)):
        raise TypeError('image must be a list or numpy array')

    blobs_flat = flatten(blobs)  # flatten nested list

    if not all((isinstance(blob, bc.Blob) for blob in blobs_flat)):
        raise TypeError('blob_list must only contain blobs')

    im_copy = im.copy()  # create copy of image to return
    contours = [blob.cv_contour for blob in blobs_flat]

    # draw contours on the copied image
    cv.drawContours(im_copy, contours, -1, (0, 255, 0), 2, cv.LINE_8)

    return im_copy


def get_maxima(image, distance=20, threshold=0.8):
    """
    Returns local maxima in an image

    Parameters
    ----------
    image : numpy ndarray or Blob instance
        Image/Blob to find maxima for

    distance : int
        Minimum pixel distance between maxima

    threshold : float or int
        Minimum relative intensity threshold for maxima

    Returns
    -------
    local_max_coords : list
        A list of x,y coordinates of local maxima
    """
    if not isinstance(distance, int):
        raise TypeError('distance must be an int')

    if not isinstance(threshold, (int, float)):
        raise TypeError('threshold must be int or float')

    if not (distance >= 0):
        raise IndexError('distance must be >= 0')

    if threshold > 1 or threshold < 0:
        raise IndexError('threshold must be 0 <= threshold <= 1')

    if isinstance(image, np.ndarray):
        if image.ndim > 2:
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        local_max_thresh = np.mean(image) * threshold
        local_max_coords = peak_local_max(image,
                                          min_distance=distance,
                                          threshold_abs=local_max_thresh)
        local_max_coords = [[x, y] for y, x in local_max_coords]  # y,x -> x,y
        return local_max_coords

    elif isinstance(image, bc.Blob):
        local_max_thresh = image.pixel_intensity_percentile(threshold * 100)
        local_max_coords = peak_local_max(image.image_masked,
                                          min_distance=distance,
                                          threshold_abs=local_max_thresh)
        local_max_coords = [[x, y] for y, x in local_max_coords]  # y,x -> x,y
        return local_max_coords

    else:
        raise TypeError('image must be numpy array or Blob instance')


def get_contours(image, thresh_min, thresh_max=255, thresh_step=10):
    """
    Returns list of contours in image

    Parameters
    ----------
    image : numpy ndarray
        Image to find contours in

    thresh_min : int
        Minimum threshold

    thresh_max : int
        Maximum threshold

    thresh_step: int
        Step between min and max threshold

    Returns
    -------
    contours : list of numpy ndarrays
        A list of contours
    """
    if not isinstance(image, (list, np.ndarray)):
        raise TypeError('image must be a list or numpy array')

    if not isinstance(thresh_min, int):
        raise TypeError('thresh_min must be an int')

    if not isinstance(thresh_max, int):
        raise TypeError('thresh_max must be an int')

    if not isinstance(thresh_step, int):
        raise TypeError('thresh_step must be an int')

    if not (thresh_min >= 0 and thresh_min <= thresh_max):
        raise IndexError('thresh_min must be 0 <= thresh_min <= thresh_max')

    if not (thresh_max >= thresh_min and thresh_max <= 255):
        raise IndexError('thresh_max must be thresh_min <= thresh_max <= 255')

    if not (thresh_step >= 1 and thresh_step <= thresh_max):
        raise IndexError('thresh_step must be 1 <= thresh_step <= thresh_max')

    im_thresh = [cv.threshold(image, i, 255, cv.THRESH_BINARY)[1]
                 for i in range(thresh_min, thresh_max, thresh_step)]

    contours_nested = [cv.findContours(thresh,
                                       cv.RETR_EXTERNAL,
                                       cv.CHAIN_APPROX_NONE)[0]
                       for thresh in im_thresh]

    contours = [contour for contours in contours_nested
                for contour in contours]

    return contours


def segment_contours(binary_image, min_distance=10):
    """
    Segments binary image and returns list of contours

    Parameters
    ----------
    binary_image : numpy ndarray
        Image to find contours in

    min_distance : int
        Minimum distance between peaks of segmented image

    Returns
    -------
    contours : list of numpy ndarrays
        A list of contours
    """
    if not isinstance(binary_image, np.ndarray):
        raise TypeError('binary_image must be a numpy array')

    if not isinstance(min_distance, int):
        raise TypeError('min_distance must be an int')

    if not (min_distance >= 0):
        raise IndexError('min_distance must be >= 0')

    # apply distance transform
    distance = ndi.distance_transform_edt(binary_image)

    # find maximum coordinates and label on mask
    coords = peak_local_max(distance,
                            threshold_rel=0.5,
                            min_distance=min_distance,
                            footprint=np.ones((3, 3)),
                            labels=binary_image)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True

    # perform watershed segmentation
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=binary_image)

    # get contours of labeled points
    contours = []
    for i in range(1, np.max(labels)+1):
        binary = np.zeros(labels.shape, dtype="uint8")
        binary[np.where(labels == i)] = 255
        contour, _ = cv.findContours(binary,
                                     cv.RETR_EXTERNAL,
                                     cv.CHAIN_APPROX_NONE)
        contours.append(contour[0])

    return contours


def organize_contours(contours, coords, image, min_distance):
    """
    Organizes contours into list based on image coordinates

    Parameters
    ----------
    contours : nested list or numpy ndarray
        each inner list contains contour points
        contours to organize

    coords : nested list or numpy ndarray
        each inner list is an image coordinate
        coordinates to organize contours by

    image : numpy ndarray
        image to create Blob instance with

    min_distance : int
        Minimum distance between peaks of segmented contours

    Returns
    -------
    contour_list : nested list
        each inner list contains Blob instances
        Blob instances are organized by coordinates
    """
    if not isinstance(contours, (list, tuple, np.ndarray)):
        raise TypeError('contours must be a list, tuple, or numpy array')

    if not isinstance(coords, (list, tuple, np.ndarray)):
        raise TypeError('coords must be a list, tuple, or numpy array')

    if not isinstance(image, np.ndarray):
        raise TypeError('image must be a numpy array')

    if not isinstance(min_distance, int):
        raise TypeError('min_distance must be an int')

    if not (min_distance >= 0):
        raise IndexError('min_distance must be >= 0')

    # initalize empty list that will hold contours
    contour_list = [[] for pt in coords]

    # initalize list that contains index of points in contour_list
    # unique_points will contain single point idxs (ex. [2]) and
    # multiple point idxs (ex. [2, 5]) in case a contour always
    # contains multiple maxima
    unique_points = [[i] for i in range(len(coords))]

    # for each original contour found
    for contour in contours:
        maxima, idxs = maxima_filter(contour, coords)
        if idxs is not None:

            # if contour contains one maxima, append to idx in list
            if len(idxs) == 1:
                contour_list[idxs[0]].append(bc.Blob(contour, image))

            # if contour contains more than one maxima, segment contour
            else:
                seg_binary = np.zeros(image.shape, dtype="uint8")
                cv.fillPoly(seg_binary, pts=[contour], color=(255, 255, 255))
                seg_contours = segment_contours(seg_binary, min_distance)
                # for each segmented contour
                for sc in seg_contours:
                    sc_maxima, sc_idxs = maxima_filter(sc, coords)
                    if sc_idxs is not None:

                        # if segmented contour contains one maxima,
                        # append to idx in list
                        if len(sc_idxs) == 1:
                            contour_list[sc_idxs[0]].append(bc.Blob(sc, image))

                        # if segmented contour contains more than one maxima
                        else:
                            # new point (multiple idxs) and blob
                            sc_idxs = list(sc_idxs)
                            sc_blob = bc.Blob(sc, image)

                            # if new point not already in unqiue_points,
                            # add to unique_points and
                            # add empty list to contour_list
                            if sc_idxs not in unique_points:
                                unique_points.append(sc_idxs)
                                contour_list.append([])

                            # get index and add new blob to contour_list
                            sc_idx = unique_points.index(sc_idxs)
                            contour_list[sc_idx].append(sc_blob)

    return contour_list


def maxima_filter(contour, local_maxima):
    """
    Returns maxima inside contour if contour contains 1 or more maxima

    Parameters
    ----------
    contour : list
        Contains contour points

    local_maxima : list
        Contains x,y coordinates of local maxima

    Returns
    -------
    maxima, idxs : tuple of lists
        maxima - list containing x,y coordinates of maxima inside contour
        idxs - list of indices of maxima in local_maxima list
    """
    if not isinstance(contour, (list, np.ndarray)):
        raise TypeError('contour must be list or numpy array')

    if not isinstance(local_maxima, list):
        raise TypeError('local_maxima must be list')

    if not ((len(np.shape(contour)) == 2 and np.shape(contour)[1] == 2) or
            (len(np.shape(contour)) == 3 and np.shape(contour)[2] == 2)):
        raise IndexError('contour should have shape (x,2) or (x,1,2)')

    if (len(np.shape(local_maxima)) != 2 or
            np.shape(local_maxima)[1] != 2):
        raise IndexError('local_maxima should have shape (x,2)')

    # points is list of booleans
    # True in each index that maxima is inside contour
    points = np.array([cv.pointPolygonTest(np.array(contour),
                                           (int(maxima[0]), int(maxima[1])),
                                           False)
                       for maxima in local_maxima])

    if np.sum(points == 1) >= 1:  # if maxima found
        idxs = np.where(points == 1)  # points are true
        maxima = np.array(local_maxima)[idxs]
        return maxima, idxs[0]

    # return None if no maxima found
    return None, None


def blob_filter(blob, filters):
    """
    Returns True if blob conforms to specified filters

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
    if not isinstance(blob, bc.Blob):
        raise TypeError('blob must be object of class Blob')

    if not isinstance(filters, list):
        raise TypeError('filters must be a list')

    if len(np.shape(filters)) != 2 or np.shape(filters)[1] != 3:
        raise IndexError('filters should have shape (x,3)')

    for filt in filters:

        if filt[1] is not None:
            if not (eval('blob.' + filt[0]) >= filt[1]):
                return False

        if filt[2] is not None:
            if not (eval('blob.' + filt[0]) <= filt[2]):
                return False

    return True


def similar_filter(blob_list, params, num=2):
    """
    Returns most similar blobs in a list of blobs

    Description
    -----------
    Blobs are compared to one another based on a number of
    specified parameters. The blob list is split into multiple
    lists of similar blobs. The list with the highest
    number of similar blobs is returned.

    Parameters
    ----------
    blob_list : list of Blob objects (class Blob)
        Contains blobs which will be analyzed for similarity
        Blob order in list matters as each blob is evaluated
        against the previous one

    params : nested list
        Each inner list contains:
            1st index - blob parameter
            2nd index - percent threshold for similarity

    num : int, optional (default 2)
        Must be >= 1
        Minimum value of similar blobs that are needed to both
        analyze similarity and return the similar blobs

    Returns
    -------
    sim_blobs - list of Blob objects
        Largest collection of most similar blobs in blob_list

    Example
    -------
    params = [['area', 0.2], ['circularity', 0.1]]
    blob_filter(blob_list, params) will parse blob_list into lists
    containing blobs that are similar in area (within 20%) and
    circularity (within 10%). The list with the highest number
    of similar blobs is returned.
    """
    if not isinstance(blob_list, list):
        raise TypeError('blob_list must be a list')

    if not all((isinstance(blob, bc.Blob) for blob in blob_list)):
        raise TypeError('blob_list must only contain blobs')

    if not isinstance(params, list):
        raise TypeError('params must be a list')

    if len(np.shape(params)) != 2 or np.shape(params)[1] != 2:
        raise IndexError('params should have shape (x,2)')

    if not isinstance(num, int):
        raise TypeError('num must be an int')

    if num < 1:
        raise IndexError('num must be >= 1')

    # First check that blob_list is long enough to analyze
    if len(blob_list) >= num:
        sim_list = [[]]
        sim_idx = 0
        last_blob = None

        for blob in blob_list:
            if last_blob is not None:  # if not the first blob in list
                filters = []

                # Create filters to be passed into blob_filter()
                # Filters are based on previous blob in blob_list
                # and used to determine if the current blob is
                # similar enough to the previous blob
                for i in range(len(params)):
                    val = eval('last_blob.' + params[i][0])
                    val_range = val * params[i][1]
                    val_min = val - val_range
                    val_max = val + val_range
                    filters.append([params[i][0], val_min, val_max])

                # check if current blob is similar to previous blob
                # and add nested list to sim_list if NOT similar
                if not blob_filter(blob, filters):
                    sim_list.append([])
                    sim_idx += 1

            # Append blob to appropiate location in sim_list
            # Same list as previous blob if similar
            # New inner list if NOT similar
            sim_list[sim_idx].append(blob)
            last_blob = blob

        # Get the nested list of similar blobs with highest number of blobs
        sim_len = [len(sim) for sim in sim_list]
        sim_max_idx = [idx for idx, item in enumerate(sim_len)
                       if item == max(sim_len)]
        sim_blobs = sim_list[sim_max_idx[0]]  # returns first index if tie

        # Final check that similar blobs are above the required number
        if len(sim_blobs) >= num:
            return sim_blobs

    # Return None if not enough similar blobs are found
    return None


def outlier_filter(blob_list, params):
    """
    Removes outliers from list of blobs in order of params

    Description
    -----------
    For each blob parameter, zscores are calculated for the blobs
    in the list. If a blob is outside the specified zscore and
    outside a specified percentage of the mean, it is
    identified as an outlier and removed from the list

    Parameters
    ----------
    blob_list : list of Blob objects (class Blob)
        Outliers will be removed from this list

    params : nested list
        Each inner list contains:
            1st index - blob parameter
            2nd index - percent threshold for removing outlier
            3rd index - zscore threshold for removing outlier

    Returns
    -------
    blob_copy : list
        Copy of blob_list without the outliers

    Example
    -------
    params = [['area', 0.2, 1], ['circularity', 0.1, 2]]
    blob_filter(blob_list, params) will remove blobs from blob_list
    if the blob has an area outside 20% of the blob_list mean and has
    a zscore above 1 OR if the blob has a circularity outside 10% of
    the blob_list mean and has a zscore above 2. Returns blob_list with
    the outliers removed
    """
    if not isinstance(blob_list, list):
        raise TypeError('blob_list must be a list')

    if not all((isinstance(blob, bc.Blob) for blob in blob_list)):
        raise TypeError('blob_list must only contain blobs')

    if not isinstance(params, list):
        raise TypeError('params must be a list')

    if len(np.shape(params)) != 2 or np.shape(params)[1] != 3:
        raise IndexError('params should have shape (x,3)')

    # create copy of list as numpy array
    blob_copy = np.array(blob_list.copy())

    for param in params:
        vals = [eval('blob.' + param[0]) for blob in blob_copy]
        zscores = np.array(zscore(vals))  # calculate zscores for param
        mean = np.mean(vals)  # calculate mean of param

        # Calculate safe zone (not outlier) if blob is within a specified
        # percentage of the mean of the blobs
        safe = [mean - (mean * param[1]), mean + (mean * param[1])]

        # Remove outliers if blob not within safe zone or specified zscore
        out_coords = np.where(((zscores >= -param[2]) & (zscores <= param[2]))
                              | ((vals >= safe[0]) & (vals <= safe[1])))
        blob_copy = blob_copy[out_coords]

    return list(blob_copy)


def blob_best(blob_list, criteria):
    """
    Returns the blob with the highest score out of a list of blobs

    Descripton
    ----------
    For each criteria, the blobs are ranked on how closely they meet
    it. For a blob list of length 5, the blob that most closely matches
    the criteria would get a score of 4, and the blob that's farthest
    from the criteria would get a score of 0. The scores would then be
    multiplied by a factor if specified. A blob gets a score for each
    criteria and the blob with the highest overall score is returned.

    Parameters
    ----------
    blob_list : list of Blob objects (class Blob)

    criteria : nested list
        Each inner list contains:
            1st index - blob parameter
            2nd index - ideal parameter value; or 'min' or 'max'
            3rd index - criteria factor to multiply score by

    Returns
    -------
    blob : Blob object (class Blob)
        Blob with highest score in list

    Example
    -------
    criteria = [['area', 50, 1], ['circularity', 'max', 2]]
    blob_filter(blob_list, criteria) will first rank blobs by how
    close they are to an area of 50 pixels. It will then rank blobs
    by how high their circularity is, with the score for circularity
    being multiplied by 2. The blob with the highest combined score
    will be returned.
    """
    if not isinstance(blob_list, list):
        raise TypeError('blob_list must be a list')

    if not all((isinstance(blob, bc.Blob) for blob in blob_list)):
        raise TypeError('blob_list must only contain blobs')

    if not isinstance(criteria, list):
        raise TypeError('params must be a list')

    if len(np.shape(criteria)) != 2 or np.shape(criteria)[1] != 3:
        raise IndexError('criteria should have shape (x,3)')

    if len(blob_list) == 0:
        return None

    if len(blob_list) == 1:
        return blob_list[0]

    # initial score of 0 for each blob
    score = np.zeros(len(blob_list))

    for crit in criteria:
        # get blob values for the criteria
        vals = np.array([eval('blob.' + crit[0]) for blob in blob_list])

        # vals_unique is a list of unique values in vals
        # rank is how well blob fits criteria (highest score = closest)
        # rank is automatically sorted low -> high so needs to be
        # reversed in 'min' and ideal value conditions (ex. [1,0,2] -> [1,2,0])
        if crit[1] == 'min':
            vals_unique, rank = np.unique(vals, return_inverse=True)
            rank = abs(rank-(len(vals_unique)-1))

        elif crit[1] == 'max':
            vals_unique, rank = np.unique(vals, return_inverse=True)

        elif type(crit[1]) == int or type(crit[1]) == float:
            diff = abs(vals - crit[1])
            vals_unique, rank = np.unique(diff, return_inverse=True)
            rank = abs(rank-(len(vals_unique)-1))

        rank = np.multiply(rank, crit[2])  # multiple rank by factor
        score = np.add(score, rank)  # add rank to overall score

    # Find idx of blob with highest score
    max_score = max(score)
    max_idx = list(score).index(max_score)

    # Return highest scoring blob
    return blob_list[max_idx]


def final_blobs_filter(blobs):
    """
    Finds blobs that may overlap and removes
    unncessary blobs from list

    Descripton
    ----------
    This is similar to making a tree where each parent blob contains
    children blobs which are inside of the parent blob. However, only
    the children for each blob are determined. If a parent blob has 1
    child, then the child is removed as the parent encomposses the larger
    overall blob. If the parent contains 2 or more chldren, the parent is
    removedas the child blobs better seperate out blobs in the image.

    Parameters
    ----------
    blobs : list of Blob objects (class Blob)
        Unneccessary blobs will be removed

    Returns
    -------
    final_blobs : list
        Copy of blobs without unnessessary blobs
    """
    if not isinstance(blobs, list):
        raise TypeError('blobs must be a list')

    if not all((isinstance(blob, bc.Blob) for blob in blobs)):
        raise TypeError('blobs must only contain blobs')

    # get blob centers
    blob_centers = [[int(blob.centroid_xy[0]), int(blob.centroid_xy[1])]
                    for blob in blobs]

    # sort by area - largest first
    blob_area = [blob.area_filled for blob in blobs]
    blob_rank = np.argsort(-np.array(blob_area))
    blob_centers = np.array(blob_centers)[blob_rank].astype(np.uint16)
    final_blobs = np.array(blobs)[blob_rank]

    # Initialize values for for loop
    children = [[]]  # will be changed promptly
    num_iter = 0  # number of iterations
    first_flag = True
    while ((max([len(x) for x in children]) != 0 or first_flag is True)
           and num_iter <= 10):

        # only run after first go (if children blobs found)
        if first_flag is False:
            blobs_to_remove = []
            for i in range(len(children)):

                # if only one child, keep parent
                # parent encompasses more and is likely correct
                if len(children[i]) == 1:
                    blobs_to_remove.append(children[i][0])

                # if >= 2 children, keep children
                # parent encompasses multiple blobs and is likely incorrect
                elif len(children[i]) >= 2:
                    blobs_to_remove.append(i)

            final_blobs = np.delete(final_blobs, blobs_to_remove)
            blob_centers = np.array([list(pt) for i, pt
                                     in enumerate(blob_centers)
                                     if i not in blobs_to_remove]
                                    ).astype(np.uint16)

        # Find children for each blob
        # each idx in children corresponds to blob
        children = [[] for i in final_blobs]

        # i is index of potential child blob
        # j is index of parent blob
        # In for loop:
        # i is blobs in order of desending area
        for i in range(len(final_blobs)):
            # j is blobs larger than blobs[i] in ascending order
            for j in range(i-1, -1, -1):
                # Determine if blobs[i] is in blobs[j] and
                # break if found (smallest blob containing blobs[i]
                # is found)
                if cv.pointPolygonTest(final_blobs[j].cv_contour,
                                       blob_centers[i],
                                       False) == 1:

                    children[j].append(i)
                    break

        first_flag = False  # first pass over
        num_iter += 1

    return final_blobs
