import cv2 as cv
import numpy as np
import pandas as pd
from skimage import filters
import blob_args as ba
import blob_class as bc
import blob_detection_2 as bd
import contrast_functions as cf
import nucleus_contour as nc
import param_output as po


def main():

    # read in image and save
    im = cv.imread('example_images/ex11.tif')
    cv.imwrite('01_original_image.jpg', im)

    # scale image so that at least 0.35% of pixels
    # are at the min and max (0 and 255)
    im_scaled = cf.percentile_rescale(im, 0.35, 99.65)

    # convert image to gray scale
    im_gray = cv.cvtColor(im_scaled, cv.COLOR_BGR2GRAY)

    cv.imwrite('02_scaled_gray_image.jpg', im_gray)

    # blur image and save
    im_blur = cv.GaussianBlur(im_gray, (15, 15), 0)
    im_blur = cf.percentile_rescale(im_blur, 0, 100)
    cv.imwrite('03_blurred_image.jpg', im_blur)

    # Get blob of nucleus
    nuc_blur = bd.blob_nuclei(im_blur)[0]

    # save image with all blob contours
    cv.imwrite('04_nucleus_contour.jpg', bd.blob_im(im_scaled, [nuc_blur]))

    # find maxima in nucleus
    median = int(nuc_blur.pixel_intensity_percentile(50))
    nuc_mask = nuc_blur.image_masked
    nuc_fill = nuc_mask.copy()
    nuc_fill[nuc_fill < median] = median
    nuc_fill = cf.percentile_rescale(nuc_fill, 0, 100) # maybe remove

    edges = filters.sobel(nuc_fill)
    edges = cf.percentile_rescale(edges, 0, 100)

    nuc_fill_blob = bc.Blob(nuc_blur.cv_contour, nuc_fill)

    thresh = int(nuc_blur.pixel_intensity_percentile(80))
    maxima = bd.get_maxima(nuc_fill_blob, thresh)

    # create and save image showing maxima
    im_maxima = im_scaled.copy()
    for coordinate in maxima:
        cv.circle(im_maxima, (coordinate), 2, (0, 0, 255), 2)
    cv.imwrite('05_peak_local_maxima.jpg', im_maxima)

    # set minimum threshold to find contours
    contours, _ = bd.get_contours(nuc_blur.image_masked, thresh_min=median, thresh_step=5)

    # get list of blobs organized by maxima
    org_contours = bd.organize_contours(contours, maxima)

    # save image with all blob contours
    cv.imwrite('06_maxima_blobs.jpg', bd.blob_im(im_scaled, contours))

    # get best cnts for each maxima/combo
    cnts_best = bd.best_contours(org_contours, edges)
    cv.imwrite('07_cnts_best.jpg', bd.blob_im(im_scaled, cnts_best))
    
    # get final cnts for each maxima/condo with no overlap
    cnts_final = bd.contour_remove_overlap(cnts_best, edges)
    cv.imwrite('08_cnts_final.jpg', bd.blob_im(im_scaled, cnts_final))
    
    # make blobs of final cnts
    blobs_final = [bc.Blob(cnt, im_blur) for cnt in cnts_final]

    # convert blobs to original image instead of blurred
    final_blobs = [bc.Blob(blob.cv_contour, im_scaled) for blob in blobs_final]

    # get parameters of final blobs
    params = po.get_params(final_blobs)
    params.to_csv("blob_params.csv")


if __name__ == '__main__':
    main()