import cv2 as cv
import numpy as np
import blob_class as bc
import blob_detection as bd

# read in image and save
im = cv.imread('ex6.tif')
cv.imwrite('1_original_image.jpg', im)

# convert image to gray scale
im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

# blur image and save
im_blur = cv.GaussianBlur(im_gray,(15,15),0)
cv.imwrite('2_blurred_image.jpg', im_blur)

# REPLACE WITH HENRY CODE
ret, im_binary = cv.threshold(im_blur, 25, 255, cv.THRESH_BINARY)
contours, hierarchy = cv.findContours(im_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
contour = max(contours, key=cv.contourArea)

# convert nucleus into Blob object
nuc_blur = bc.Blob(contour, im_blur)

# find maxima in nucleus
local_max_coords = bd.get_maxima(nuc_blur, 5, 0.8)

# create and save image showing maxima
im_maxima = im.copy()
for coordinate in local_max_coords:
    cv.circle(im_maxima, (coordinate), 2, (255,0,0), 2)
cv.imwrite('3_peak_local_maxima.jpg', im_maxima)

# set minimum threshold to find contours
min_thresh = int(nuc_blur.pixel_intensity_percentile(80))

# find contours
contours = bd.get_contours(im_blur, min_thresh, thresh_step=3)

# initalize empty list that will hold contours
contour_list = [[] for pt in local_max_coords]

# initalize list that contains index of points in contour_list
# unique_points will contain single point idxs (ex. [2]) and 
# multiple point idxs (ex. [2, 5]) in case a contour always
# contains multiple maxima
unique_points = [[i] for i in range(len(local_max_coords))]

if len(contours) > 0:
    
    # for each original contour found
    for contour in contours:
        maxima, idxs = bd.maxima_filter(contour, local_max_coords)
        if idxs is not None:
            
            # if contour contains one maxima, append to idx in list
            if len(idxs) == 1:
                contour_list[idxs[0]].append(bc.Blob(contour, im_blur))
            
            # if contour contains more than one maxima, segment contour
            else: # if len(idxs) > 1
                seg_binary = np.zeros(im_gray.shape, dtype="uint8")
                cv.fillPoly(seg_binary, pts=[contour], color=(255,255,255))
                seg_contours = bd.segment_contours(seg_binary, min_distance=5)
                # for each segmented contour
                for sc in seg_contours:
                    sc_maxima, sc_idxs = bd.maxima_filter(sc, local_max_coords)
                    if sc_idxs is not None:
                        
                        # if segmented contour contains one maxima, append to idx in list
                        if len(sc_idxs) == 1:
                            contour_list[sc_idxs[0]].append(bc.Blob(sc, im_blur))
                            
                        # if segmented contour contains more than one maxima 
                        else:
                            # new point (multiple idxs) and blob
                            sc_idxs = list(sc_idxs)
                            sc_blob = bc.Blob(sc, im_blur) #MAY NEED TO SWITCH BACK TO im
                            
                            # if new point not already in unqiue_points,
                            # add to unique_points and add empty list to contour_list
                            if sc_idxs not in unique_points:
                                unique_points.append(sc_idxs)
                                contour_list.append([])
                                
                            # get index and add new blob to contour_list
                            sc_idx = unique_points.index(sc_idxs)
                            contour_list[sc_idx].append(sc_blob)

# save image with all blob contours
cv.imwrite('4_maxima_blobs.jpg', bd.blob_im(im, contour_list))

# create filters for blob
filters = [['area', 10, None], # 25 for ex111.tif
           ['ellipse_fit_residual_mean', None, 1],
           ['pixel_kurtosis', None, 0]]

# filter blobs based on filters list
blob_list = []
for contours in contour_list:
    blob_list.append([x for x in contours if bd.blob_filter(x, filters)])

# save image with filtered contours
cv.imwrite('5_filtered_blob.jpg', bd.blob_im(im, blob_list))

# create params to get blobs most simliar to each other
params = [['pixel_intensity_percentile(10)', 0.2]]

# filter blobs based on similarity
sim_blobs = [bd.similar_filter(blobs, params, 2) for blobs in blob_list]
sim_blobs = [blobs for blobs in sim_blobs if blobs is not None]

# save image with most similar blobs contours for each unique_point
cv.imwrite('6_similar_blobs.jpg', bd.blob_im(im, sim_blobs))

# create out_filter list to filter out outlier blobs
out_filter = [['pixel_intensity_percentile(10)', 0.2, 1]]

# filter out outliers
no_outs = [bd.outlier_filter(blobs, out_filter) for blobs in sim_blobs] 

# save image with blobs that are not outliers
cv.imwrite('7_no_outlier_blobs.jpg', bd.blob_im(im, no_outs))

# create criteria for determining best blob
criteria = [['area_filled', 'max', 1],
            ['roughness_surface', 'min', 1]]

# get blob with highest score for each unique_point
blobs_best = [bd.blob_best(blobs, criteria) for blobs in no_outs]

# now need to make sure blobs don't overlap, get blob centers
blob_centers = [[int(blob.centroid_xy[0]), int(blob.centroid_xy[1])] for blob in blobs_best]
blob_centers = [list(y) for y in set([tuple(x) for x in blob_centers])] # remove duplicates

# initialize list to group blobs based on blob centroids
blob_groups = [[] for pt in blob_centers]
for blob in blobs_best:
    maxima, idxs = bd.maxima_filter(blob.contour, blob_centers)
    # for each center blob contains, add to blob_groups idx
    for idx in idxs:
        blob_groups[idx].append(blob)

# filter blobs_best to get non-overlapping blobs
final_blobs = blobs_best.copy()
for blobs in blob_groups:
    # if blob_groups idx contain more than one blob
    # indicates one point contains multiple blobs
    if len(blobs) > 1:
        # find blob with smallest area and keep that one
        # blob larger than smallest contain multiple center points
        # and are unneccearily large
        area = [blob.area for blob in blobs]
        min_area = min(area)
        for i in range(len(area)):
            if area[i] != min_area:
                # remove needed blobs from final_blobs
                if blobs[i] in final_blobs:
                    final_blobs.remove(blobs[i])

# save image with final blob contours
cv.imwrite('8_final_blobs.jpg', bd.blob_im(im, final_blobs))