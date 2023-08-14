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
    """
    Creates a dictionary

    Returns
    -------
    blob_params
        dictionary containing blob parameters
    """
    blob_params = {}
    for property in funcs:
        blob_params[property] = []

    return blob_params


def add_value(blob_params, blob, dec=2):
    """
    Adds values to the dictionary

    Parameters
    ----------
    blobs_params : dict
        contains blob_params

    blob : Blob object (class Blob)
        blob who's properties will be added to dictionary

    dec : int
        number of places after decimal point for properties
    """
    if not isinstance(blob_params, dict):
        raise TypeError('blob_params must be a dict')

    if not isinstance(blob, bc.Blob):
        raise TypeError('blob must be instance of Blob')

    if not isinstance(dec, int):
        raise TypeError('dec must be an int')

    for i in range(len(funcs)):
        try:
            val = eval('blob.' + funcs[i])
            blob_params[funcs[i]].append(str(np.around(val, dec)))
        except Exception as e:
            val = eval('blob.' + funcs[i] + '()')
            blob_params[funcs[i]].append(str(np.around(val, dec)))


def get_params(blobs, dec=2, nucleus=None):
    """
    Returns dataframe of blob properties

    Parameters
    ----------
    blobs : list of Blob objects (class Blob)
        Contains blobs who's parameter will be calculated

    dec : int
        number of places after decimal point for properties

    nucleus: Blob object (class Blob), optional
        A blob representing the nucleus, whose properties will be added
        as the first row in the dataframe.

    Returns
    -------
    df - Pandas dataframe
        dataframe containing properties of each blob
    """
    if not isinstance(blobs, (list, np.ndarray)):
        raise TypeError('blobs must be a list or numpy array')

    if not all((isinstance(blob, bc.Blob) for blob in blobs)):
        raise TypeError('blobs must only contain blobs')

    if not isinstance(dec, int):
        raise TypeError('dec must be an int')

    # Create dictionary
    blob_params = build_dict()

    # Get properties for nucleus if provided
    if nucleus:
        add_value(blob_params, nucleus, dec=dec)
    
    # Get properties for each blobs
    for blob in blobs:
        add_value(blob_params, blob, dec=dec)

    # Transfer dictionary to pandas dataframe
    if nucleus:
        df = pd.DataFrame(blob_params, 
                          index=['nucleus'] + ['blob' + str(i) for i in range(1, len(blobs)+1)])
    else:
        df = pd.DataFrame(blob_params, 
                          index=['blob' + str(i) for i in range(1, len(blobs)+1)])

    return df

def main():
    im = cv.imread("example_images/ex3.tif")
    im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    im_blur = cv.GaussianBlur(im_gray, (25, 25), 0)
    ret, im_thresh = cv.threshold(im_blur, 125, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(im_thresh,
                                          cv.RETR_EXTERNAL,
                                          cv.CHAIN_APPROX_NONE)
    blobs = [bc.Blob(contour, im) for contour in contours]
    df = get_params(blobs)
    df.to_csv("output.csv")


if __name__ == '__main__':
    main()
