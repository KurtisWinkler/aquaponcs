import numpy as np
from skimage import exposure


def min_max_rescale(image):
    '''
    Description:
    Improves the contrast of the image by rescaling the minimum and maximum
    pixel intensity values

    Input:
    image: array
        image to scale
    Output:
    minmax_data: array
        numpy array of pixel intensities to be saved as a .png

    '''
    if not isinstance(image, (list, np.ndarray)):
        raise TypeError('image must be a list or numpy array')

    # Rescale the min and max pixel intensity values
    image_minmax_scaled = exposure.rescale_intensity(image)

    return image_minmax_scaled


def percentile_rescale(image, min_percentile, max_percentile):
    '''
    Description:
    Improves the contrast of the image by rescaling the percentile
    pixel intensity values

    Input:
    image: array
        image to scale
    min_percentile: int or float
        minimum percentile to rescale
    max_percentile: int or float
        maximum percentile to rescale

    Output:
    scaled: numpy ndarray
        scaled version of input image
    '''
    if not isinstance(image, (list, np.ndarray)):
        raise TypeError('image must be a list or numpy array')

    if not isinstance(min_percentile, (int, float)):
        raise TypeError('min_percentile must be an int or float')

    if not isinstance(max_percentile, (int, float)):
        raise TypeError('max_percentile must be an int or float')

    # get image percentiles for rescaling
    percentiles = np.percentile(image, (min_percentile, max_percentile))

    # Rescale pixel intensity by percentile
    # in_range are pixels in range to be scaled
    # pixels out of range are scaled to min or max
    # out_range uint8 specifies output will be between 0 and 255
    scaled = exposure.rescale_intensity(image,
                                        in_range=tuple(percentiles),
                                        out_range=np.uint8)

    return scaled
