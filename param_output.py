import blob_class as bc
import pandas as pd
import numpy as np
import cv2 as cv

funcs = [
        'aspect_ratio',
        'area',
        'area_bbox',
        'area_convex',
        'area_filled',
        'axis_major_length',
        'axis_minor_length',
        'centroid_xy',
        'centroid_local',
        'centroid_weighted',
        'centroid_weighted_local',
        'circularity',
        'curvature_mean',
        'eccentricity',
        'ellipse_fit_residual_mean',
        'equivalent_diameter_area',
        'euler_number',
        'extent',
        'feret_diameter_max',
        'inertia_tensor',
        'inertia_tensor_eigvals',
        'intensity_max',
        'intensity_mean',
        'intensity_min',
        'moments',
        'moments_central',
        'moments_hu',
        'moments_normalized',
        'moments_weighted',
        'moments_weighted_central',
        'moments_weighted_hu',
        'moments_weighted_normalized',
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


def get_params(blobs):
    # Create dictionary
    blob_params = build_dict()
    
    # Get properties for each blobs
    for blob in blobs:
        add_value(blob_params, blob)

    # Transfer dictionary to pandas dataframe
    df = pd.DataFrame(blob_params,
                      index=['blob' + str(i) for i in range(1, len(blobs)+1)])
    
    return df


def main():
    im = cv.imread("ex3.tif")
    im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    im_blur = cv.GaussianBlur(im_gray, (25, 25), 0)
    ret, im_thresh = cv.threshold(im_blur, 125, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(im_thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    blobs = [bc.Blob(contour, im) for contour in contours]
    df = get_params(blobs)
    df.to_csv("output.csv")

if __name__ == '__main__':
    main()
