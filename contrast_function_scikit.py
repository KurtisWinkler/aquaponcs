import numpy as np
from skimage import exposure

def min_max_rescale(image):
    '''
    Description:
    Improves the contrast of the image by rescaling the minimum and maximum 
    pixel intensity values

    Input:
    file: str
        file path for the image
    max_rescale: int
        the pixel intensity value to rescale the current maximum up to
    min_rescale: int
        the pixel intensity value to rescale the current maximum down to
    output_name: str 
        the file path for the output .png
    Output: 
    minmax_data: array 
        numpy array of pixel intensities to be saved as a .png

    '''
    # Rescale the min and max pixel intensity values
    image_minmax_scaled = exposure.rescale_intensity(image)

    return image_minmax_scaled


def percentile_rescale(image, min_percentile, max_percentile):
    '''
    Description: 
    Improves the contrast of the image by rescaling the percentile 
    pixel intensity values

    Input:
    file: str
        file path for the image
    min_percentile: int or float
        minimum percentile to rescale 
    max_percentile: int or float
        maximum percentile to rescale

    Output: 
    scaled: numpy ndarray 
        scaled version of input image
    '''
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
