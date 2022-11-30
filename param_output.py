from IPython.display import display
from blob_class import Blob
import pandas as pd
import numpy as np
import cv2 as cv

funcs = [
        'aspect_ratio',
        'area_filled',
        'area_convex',
        'axis_major_length',
        'axis_minor_length',
        'centroid_xy',
        'circularity',
        'curvature_mean',
        'eccentricity',
        'ellipse_fit_residual_mean',
        'equivalent_diameter_area',
        'orientation',
        'perimeter_crofton',
        'perimeter_convex_hull',
        'pixel_intensity_mean',
        'pixel_intensity_median',
        'pixel_intensity_std',
        'pixel_kurtosis',
        'pixel_skew',
        'roughness_perimeter',
        'roughness_surface',
        'roundness',
        'solidity'
    ]


def build_dict():
    blob_params = {}
    for property in funcs:
        blob_params[property] = []

    return blob_params


def add_value(blob_params, blob, dec = 2):
    for i in range(len(funcs)):
        try:
            val = eval('blob.' + funcs[i])
            blob_params[funcs[i]].append(str(np.around(val, dec)))
        except:
            val = eval('blob.' + funcs[i] + '()')
            blob_params[funcs[i]].append(str(np.around(val, dec)))


def main():
    blob_params = build_dict()

    im = cv.imread("ex3.tif")
    im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    im_blur = cv.GaussianBlur(im_gray, (25, 25), 0)
    ret, im_thresh = cv.threshold(im_blur, 125, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(im_thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    # Get properties for each blobs
    for contour in contours:
        blob = Blob(contour, im)
        add_value(blob_params, blob)

    # transfer dictionary to pandas dataframe
    df = pd.DataFrame(blob_params, index=['blob' + str(i) for i in range(1, len(contours)+1)])
    display(df)
    df.to_csv("output.csv")


if __name__ == '__main__':
    main()

