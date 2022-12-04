import cv2 as cv
import blob_class as bc
import blob_detection as bd

# read in image and save
im = cv.imread('ex6.tif')
cv.imwrite('1_original_image.jpg', im)

# convert image to gray scale
im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

# blur image and save
im_blur = cv.GaussianBlur(im_gray, (15, 15), 0)
cv.imwrite('2_blurred_image.jpg', im_blur)

# REPLACE WITH HENRY CODE
ret, im_binary = cv.threshold(im_blur, 25, 255, cv.THRESH_BINARY)
contours, _ = cv.findContours(im_binary,
                              cv.RETR_EXTERNAL,
                              cv.CHAIN_APPROX_NONE)
contour = max(contours, key=cv.contourArea)

# convert nucleus into Blob object
nuc_blur = bc.Blob(contour, im_blur)

# find maxima in nucleus
local_max_coords = bd.get_maxima(nuc_blur, 5, 0.8)

# create and save image showing maxima
im_maxima = im.copy()
for coordinate in local_max_coords:
    cv.circle(im_maxima, (coordinate), 2, (0, 0, 255), 2)
cv.imwrite('3_peak_local_maxima.jpg', im_maxima)

# set minimum threshold to find contours
min_thresh = int(nuc_blur.pixel_intensity_percentile(80))

# find contours
contours = bd.get_contours(im_blur, min_thresh, thresh_step=3)

# get list of blobs organized by maxima
contour_list = bd.organize_contours(contours,
                                    local_max_coords,
                                    im_blur,
                                    min_distance=5)

# save image with all blob contours
cv.imwrite('4_maxima_blobs.jpg', bd.blob_im(im, contour_list))

# create filters for blob
filters = [['area', 10, None],
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

# save image with best blobs for each unique_point
cv.imwrite('8_best_blobs.jpg', bd.blob_im(im, blobs_best))

# remove any overlapping, unncessessary blobs
final_blobs = bd.final_blobs_filter(blobs_best)

# save image with final blob contours
cv.imwrite('9_final_blobs.jpg', bd.blob_im(im, final_blobs))
