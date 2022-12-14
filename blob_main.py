import cv2 as cv
import numpy as np
import pandas as pd
import blob_args as ba
import blob_class as bc
import blob_detection as bd
import contrast_functions as cf
import nucleus_contour as nc
import param_output as po


def main():

    # create initial filters for blobs
    # 1 - blob parameter
    # 2 - min value of parameter or None
    # 3 - max value of parameter or None
    init_filter = [['area_filled', 20, None],
                   ['circularity', 0.9, None],
                   ['ellipse_fit_residual_mean', None, 1],
                   ['pixel_kurtosis', None, 0]]

    # create filter to get blobs most simliar to each other
    # 1 - blob parameter
    # 2 - percent threshold for similarity
    sim_filter = [['pixel_intensity_percentile(10)', 0.2]]

    # create filter to remove outlier blobs
    # 1 - blob parameter
    # 2 - percent threshold for removing outlier
    # 3 - zscore threshold for removing outlier
    out_filter = [['pixel_intensity_percentile(10)', 0.2, 1]]

    # create criteria for determining best blob
    # 1 - blob parameter
    # 2 - ideal parameter value; or 'min' or 'max'
    # 3 - criteria factor to multiply score by
    best_filter = [['area_filled', 'max', 1],
                   ['roughness_surface', 'min', 1]]

    # get input arguments
    args = ba.get_args()

    # read in image and save
    im = cv.imread(args.file_name)
    cv.imwrite('01_original_image.jpg', im)

    # scale image so that at least 0.35% of pixels
    # are at the min and max (0 and 255)
    im_scaled = cf.percentile_rescale(im, 0.35, 99.65)

    # convert image to gray scale
    im_gray = cv.cvtColor(im_scaled, cv.COLOR_BGR2GRAY)

    cv.imwrite('02_scaled_gray_image.jpg', im_gray)

    # blur image and save
    im_blur = cv.GaussianBlur(im_gray, (15, 15), 0)
    cv.imwrite('03_blurred_image.jpg', im_blur)

    # Get contour of nucleus
    nuc_contour = nc.nucleus_contour(im_gray,
                                     num_iter=10,
                                     smoothing=5)

    # convert nucleus into Blob object
    nuc_blur = bc.Blob(nuc_contour, im_blur)

    # save image with all blob contours
    cv.imwrite('04_nucleus_contour.jpg', bd.blob_im(im_scaled, [nuc_blur]))

    # find maxima in nucleus
    local_max_coords = bd.get_maxima(image=nuc_blur,
                                     distance=args.min_distance,
                                     threshold=args.min_thresh_maxima)

    # create and save image showing maxima
    im_maxima = im_scaled.copy()
    for coordinate in local_max_coords:
        cv.circle(im_maxima, (coordinate), 2, (0, 0, 255), 2)
    cv.imwrite('05_peak_local_maxima.jpg', im_maxima)

    # set minimum threshold to find contours
    min_thresh = int(nuc_blur.pixel_intensity_percentile(
                     args.min_thresh_contours * 100))

    # find contours
    contours = bd.get_contours(image=im_blur,
                               thresh_min=min_thresh,
                               thresh_step=args.thresh_step)

    # get list of blobs organized by maxima
    contour_list = bd.organize_contours(contours,
                                        local_max_coords,
                                        im_blur,
                                        min_distance=args.min_distance)
    # remove empty lists
    contour_list = [contour for contour in contour_list if contour]

    # save image with all blob contours
    cv.imwrite('06_maxima_blobs.jpg', bd.blob_im(im_scaled, contour_list))

    # filter blobs based on filters list
    if args.no_init_filter is False:
        blob_list = []
        for contours in contour_list:
            blob_list.append([x for x in contours
                              if bd.blob_filter(x, init_filter)])

        # save image with filtered contours
        cv.imwrite('07_filtered_blob.jpg', bd.blob_im(im_scaled, blob_list))
    else:
        blob_list = contour_list

    # filter blobs based on similarity
    if args.no_sim_filter is False:
        sim_blobs = [bd.similar_filter(blobs, sim_filter, 2)
                     for blobs in blob_list]
        sim_blobs = [blobs for blobs in sim_blobs if blobs is not None]

        # save image with most similar blobs contours for each unique_point
        cv.imwrite('08_similar_blobs.jpg', bd.blob_im(im_scaled, sim_blobs))
    else:
        sim_blobs = blob_list
        sim_blobs = [blobs for blobs in sim_blobs if blobs]

    # filter out outliers
    if args.no_out_filter is False:
        no_outs = [bd.outlier_filter(blobs, out_filter) for blobs in sim_blobs]

        # save image with blobs that are not outliers
        cv.imwrite('09_no_outlier_blobs.jpg', bd.blob_im(im_scaled, no_outs))
    else:
        no_outs = sim_blobs

    # get blob with highest score for each unique_point
    blobs_best = [bd.blob_best(blobs, best_filter) for blobs in no_outs]

    # save image with best blobs for each unique_point
    cv.imwrite('10_best_blobs.jpg', bd.blob_im(im_scaled, blobs_best))

    # remove any overlapping, unncessessary blobs
    final_blobs = bd.final_blobs_filter(blobs_best)

    # save image with final blob contours
    cv.imwrite('11_final_blobs.jpg', bd.blob_im(im_scaled, final_blobs))

    # convert blobs to original image instead of blurred
    final_blobs = [bc.Blob(blob.cv_contour, im_scaled) for blob in final_blobs]

    # get parameters of final blobs
    params = po.get_params(final_blobs)
    params.to_csv("blob_params.csv")


if __name__ == '__main__':
    main()
