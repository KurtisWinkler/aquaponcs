import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import data, filters, morphology
from skimage.measure import shannon_entropy
from scipy.spatial import distance as sc_distance
from skimage.feature import peak_local_max, canny
from skimage.segmentation import watershed
import torch
import torchvision.ops.boxes as bops
import blob_class as bc
import contrast_functions as cf


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

    if not all((isinstance(blob, (bc.Blob, np.ndarray)) for blob in blobs_flat)):
        raise TypeError('blob_list must only contain blobs')

    im_copy = im.copy()  # create copy of image to return
    
    if all((isinstance(blob, bc.Blob) for blob in blobs_flat)):
        cnts = [blob.cv_contour for blob in blobs_flat]
    else:
        cnts = blobs_flat
    # draw contours on the copied image
    cv.drawContours(im_copy, cnts, -1, (0, 255, 0), 2, cv.LINE_8)

    return im_copy


def blob_nuclei(image, min_percent_area=10):
    """
    Finds cell nuclei in an image

    Parameters
    ----------
    image : image matrix
        input image

    min_percent_area : int or float
        Minimum percent of image area to be considered
        an actual nuclei

    Returns
    -------
    blob_nuclei: list of Blob objects
        A list of blob nuclei found in image
    """

    # find edges in the image
    edges = filters.sobel(image)
    edges = cf.percentile_rescale(edges, 0.35, 99.65)

    # make binary image
    thresh = filters.threshold_li(edges)
    _ , im_binary = cv.threshold(edges, thresh, 255, cv.THRESH_BINARY)
    
    # find and sort contours by area - largest first
    contours, _ = cv.findContours(im_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    
    # find min_area
    shape = image.shape
    ax1 = shape[0] * min_percent_area / 100
    ax2 = shape[1] * min_percent_area / 100
    min_area = ax1 * ax2
    
    # get blob_nuclei which are above correct size
    # have to make Blob object to get pixel area
    blob_nuclei = []
    for cont in contours:
        # create Blob object
        blob = bc.Blob(cont, image)

        # append blob if large enough
        if blob.area_filled > min_area:
            blob_nuclei.append(blob)

        # else break as next blob would be smaller
        else:
            break
        
    return blob_nuclei


def get_maxima(blob, threshold):

    lap = filters.laplace(blob.image_masked)
    
    lap_coords = np.where(lap > 0)
    lap_mean = np.mean(lap[lap_coords])
    lap[lap < lap_mean] = 0
    lap[lap >= lap_mean] = 1
    lap = lap.astype(bool)
    lap = morphology.remove_small_holes(lap, area_threshold=100)
    lap = morphology.area_opening(lap, 20)

    contours, _ = cv.findContours(lap.astype(np.uint8),
                                  cv.RETR_EXTERNAL,
                                  cv.CHAIN_APPROX_NONE)

    maxima = np.array([(np.mean(contours[i], axis=0)[0].astype(int)) for i in range(len(contours))])
    
    #thresh = int(blob.pixel_intensity_percentile(threshold))

    maxima = [maxi for maxi in maxima if blob.image_masked[(maxi[1], maxi[0])] > threshold]

    return maxima


def maxima_filter(contour, local_maxima):

    if len(local_maxima) == 0:
        return None, None
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


def get_contours(image, min_length=5, thresh_min=0, thresh_max=255, thresh_step=10):
    # may want to calculate min_length (perimeter) based on min area of circle
    
    im = image.copy()
    im_thresh = []
    for i in range(thresh_max, thresh_min, -thresh_step):
        im[im >= i] = 0
        im_thresh.append(cv.threshold(im, 0.5, 255, cv.THRESH_BINARY)[1])

    all_contours = []
    maxima = []
    for thresh in im_thresh:
        contours, hierarchy = cv.findContours(thresh,
                                        cv.RETR_CCOMP,
                                        cv.CHAIN_APPROX_NONE)
        #hierachy = np.array(hierarchy[0])
        if hierarchy is not None:
            hierarchy = hierarchy[0]
            for i, row in enumerate(hierarchy):
                if len(contours[i]) >= min_length:
                    if row[2] == -1 and row[3] != -1:
                        all_contours.append(contours[i])
                        pts, idxs = maxima_filter(contours[i], maxima)
                        if idxs is None:
                            new_maxima = np.mean(contours[i], axis=0)[0].astype(int)
                            maxima.append(new_maxima)

    return all_contours, maxima


def segment_contours(binary_image, coords=None, num_peaks=None, min_distance=3):

    # apply distance transform
    distance = ndi.distance_transform_edt(binary_image)

    if coords is None and num_peaks is None:
        coords = peak_local_max(distance,
                                threshold_rel=0.5,
                                min_distance=min_distance,
                                footprint=np.ones((3, 3)),
                                labels=binary_image)
    
    elif coords is None:
        coords = peak_local_max(distance,
                                threshold_rel=0.5,
                                num_peaks=num_peaks,
                                min_distance=min_distance,
                                footprint=np.ones((3, 3)),
                                labels=binary_image)
    
    else:
        coords = np.array([[y, x] for x,y in coords])
    
    # create mask with coords
    #coords = np.array([[y, x] for x,y in coords])
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True

    # perform watershed segmentation
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=binary_image)
    #plt.imshow(labels)
    
    # get contours of labeled points
    contours = []
    for i in range(1, np.max(labels)+1):
        binary = np.zeros(labels.shape, dtype="uint8")
        binary[np.where(labels == i)] = 255
        contour, _ = cv.findContours(binary,
                                     cv.RETR_EXTERNAL,
                                     cv.CHAIN_APPROX_NONE)
        if len(contour) > 0:
            contours.append(contour[0])

    return contours


def best_contours(contours, edges):
    
    # initalize list for best contours
    best_conts = []

    for i in range(len(contours)):
        if len(contours[i]) == 1:
            best_conts.append(contours[i][0])
            
        elif len(contours[i]) > 1:
            # find means of each contour for each maxima
            mean_list = []
            for cont in contours[i]:
                pts = np.array([[pt[0][1], pt[0][0]] for pt in cont])
                edge_mean = np.mean(edges[tuple(pts.T)])
                mean_list.append(edge_mean)

            # append contour with highest edge_mean to best_conts list
            max_idx = np.argmax(mean_list)
            best_conts.append(contours[i][max_idx])
            
    return best_conts


def contour_intersect(cnt_ref, cnt_query):
    ### Thanks to Pietro Cicalese on Stack Exchange
    ## Contours are both an np array of points
    ## Check for bbox intersection, then check pixel intersection if bboxes intersect

    # first check if it is possible that any of the contours intersect
    x1, y1, w1, h1 = cv.boundingRect(cnt_ref)
    x2, y2, w2, h2 = cv.boundingRect(cnt_query)
    # get contour areas
    area_ref = cv.contourArea(cnt_ref)
    area_query = cv.contourArea(cnt_query)
    # get coordinates as tensors
    box1 = torch.tensor([[x1, y1, x1 + w1, y1 + h1]], dtype=torch.float)
    box2 = torch.tensor([[x2, y2, x2 + w2, y2 + h2]], dtype=torch.float)
    # get bbox iou
    iou = bops.box_iou(box1, box2)

    if iou == 0:
        # bboxes dont intersect, so contours dont either
        return False
    else:
        # bboxes intersect, now check pixels
        # get the height, width, x, and y of the smaller contour
        if area_ref >= area_query:
            h = h2
            w = w2
            x = x2
            y = y2
        else:
            h = h1
            w = w1
            x = x1
            y = y1

        # get a canvas to draw the small contour and subspace of the large contour
        contour_canvas_ref = np.zeros((h, w), dtype='uint8')
        contour_canvas_query = np.zeros((h, w), dtype='uint8')
        # draw the pixels areas, filled (can also be outline)
        cv.drawContours(contour_canvas_ref, [cnt_ref], -1, 255, thickness=cv.FILLED,
                         offset=(-x, -y))
        cv.drawContours(contour_canvas_query, [cnt_query], -1, 255, thickness=cv.FILLED,
                         offset=(-x, -y))

        # check for any pixel overlap
        return np.any(np.bitwise_and(contour_canvas_ref, contour_canvas_query))


def contour_remove_overlap(contours, edges):
    
    if len(contours) == 1:
        return contours

    cnts_centers = np.array([(np.mean(contours[i], axis=0)[0].astype(int))
                             for i in range(len(contours))])

    # sort by area - smallest first
    cnts_area = [cv.contourArea(cnt) for cnt in contours]
    cnts_rank = np.argsort(np.array(cnts_area))

    # find means of each contour for each maxima
    cnts_edge = []
    for cnt in contours:
        pts = np.array([[pt[0][1], pt[0][0]] for pt in cnt])
        edge_mean = np.mean(edges[tuple(pts.T)])
        cnts_edge.append(edge_mean)

    cnts_centers = np.array(cnts_centers)[cnts_rank].astype(int)
    cnts_final = np.array(contours, dtype=object)[cnts_rank]
    cnts_edge = np.array(cnts_edge)[cnts_rank]

    cnts_remove = []
    for i, cnt in enumerate(cnts_final):
        maxima, idxs = maxima_filter(cnt, cnts_centers)
        # idxs that contour contains
        if idxs is not None:
            in_idxs = [idx for idx in idxs if (idx < i and idx not in cnts_remove)]

            if len(in_idxs) > 0: # if contour contains another
                # TEST
                in_edge = np.mean(cnts_edge[in_idxs])
                out_edge = cnts_edge[i]
                
                if len(in_idxs) == 1:
                    if in_edge > out_edge:
                        cnts_remove.append(i)
                    else:
                        cnts_remove.append(in_idxs[0])
                    
                elif len(in_idxs) > 1:
                    cnts_remove.append(i)
                
                """ #ORIGINAL
                in_edge = np.mean(cnts_edge[in_idxs])
                out_edge = cnts_edge[i]

                if len(in_idxs) == 1 and in_edge > out_edge:
                    cnts_remove.append(i)  # append out contour

                # 10% buffer to consider in_edge = out_edge
                # go with smaller blobs as more likely than one large blob
                elif len(in_idxs) > 1 and 1.1 * in_edge > out_edge :
                    cnts_remove.append(i)  # append out contour

                else:
                    for idx in in_idxs:
                        if idx not in cnts_remove:
                            cnts_remove.append(idx)  # append inside contours
                """
    cnts_final = np.delete(cnts_final, cnts_remove)
    
    return cnts_final


def blob_filter(blob, filters):
    for filt in filters:

        if filt[1]:
            if not (eval('blob.' + filt[0]) >= filt[1]):
                return False

        if filt[2]:
            if not (eval('blob.' + filt[0]) <= filt[2]):
                return False

    return True


def organize_contours(contours, coords): #image): #min_distance):
    
    # sort by area - smallest first
    cnts_area = [cv.contourArea(cnt) for cnt in contours]
    cnts_rank = np.argsort(np.array(cnts_area))
    contours = np.array(contours, dtype=object)[cnts_rank]
    
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
                contour_list[idxs[0]].append(contour)

            # if contour contains more than one maxima, segment
            else:     
                # segment contour
                zeros_size = [np.max(contour)] * 2
                seg_binary = np.zeros(zeros_size, dtype="uint8")
                cv.fillPoly(seg_binary, pts=[contour], color=(255, 255, 255))
                seg_contours = segment_contours(seg_binary, num_peaks=len(maxima))#coords=maxima)

                # for each segmented contour
                for sc in seg_contours:
                    sc_maxima, sc_idxs = maxima_filter(sc, coords)
                    if sc_idxs is not None:
                        # if all maxima are the same, append full contour
                        if len(idxs) == len(sc_idxs):
                            # new point (multiple idxs)
                            sc_idxs = list(sc_idxs)

                            # if new point not already in unqiue_points,
                            # add to unique_points and
                            # add empty list to contour_list
                            if sc_idxs not in unique_points:
                                unique_points.append(sc_idxs)
                                contour_list.append([])

                            # get index and add new cnt to contour_list
                            sc_idx = unique_points.index(sc_idxs)
                            contour_list[sc_idx].append(contour) 
                        
                        # if segmented contour doesn't contain all idxs,
                        # and doesn't overlap previous contours of the other
                        # maxima, append to idx in list
                        elif len(idxs) > len(sc_idxs): #else:
                            #Find all single and multi-point contours that could have
                            #potential overlap
                            cnts_query = []
                            # all idx in idxs that are not in sc_idxs
                            qry_idxs = [idx for idx in idxs if idx not in sc_idxs]
                            for i in range(len(unique_points)):
                                # determine if qry_idxs intersect with unique_points
                                if len(set(qry_idxs).intersection(unique_points[i])) > 0:
                                    try:
                                        cnts_query.append(contour_list[i][-1])
                                    except:
                                        None

                            # determine if cnt overlaps any previous cnts
                            cnt_overlap = np.any([contour_intersect(sc, cnts_query[i])
                                                  for i in range(len(cnts_query))])
                            
                            ###TEST
                            if cnt_overlap == 0: 
                                if len(sc_idxs) == 1:
                                    contour_list[sc_idxs[0]].append(sc)
                            
                            
                                # append cnt to list if no overlap
                                elif len(sc_idxs) > 1:
                                    # new point (multiple idxs)
                                    sc_idxs = list(sc_idxs)

                                    # if new point not already in unqiue_points,
                                    # add to unique_points and
                                    # add empty list to contour_list
                                    if sc_idxs not in unique_points:
                                        unique_points.append(sc_idxs)
                                        contour_list.append([])

                                    # get index and add new blob to contour_list
                                    sc_idx = unique_points.index(sc_idxs)
                                    contour_list[sc_idx].append(sc) 
                            
    return contour_list