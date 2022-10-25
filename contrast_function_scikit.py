# scikit-image install
# conda install scikit-image

# contour detection documentation
# https://danielmuellerkomorowska.com/2020/06/27/contrast-adjustment-with-scikit-image/

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, exposure, data
from PIL import Image as im
 
# read the image
def contrast(image_name, output_name):
    image = io.imread(image_name)

    # Reports pixel intensity values on a scale of 0 to 255 for an 8-bit image
    image.max()
    image.min()

    # Option 1: Min/max rescaling (Rescale the min and max pixel intensity values)
    image_minmax_scaled = exposure.rescale_intensity(image)
    image_minmax_scaled.max()
    # 255
    image_minmax_scaled.min()
    # print(image_minmax_scaled)
    # returns an array-- how to get it to return an image? 
    # 0

    minmax_data = im.fromarray(image_minmax_scaled)
    minmax_data.save('minmax_rescaled_ex1.png')

    # Option 2: Percentile rescaling (like option 1 but uses percentiles)
    percentiles = np.percentile(image, (0.5, 99.5))
    scaled = exposure.rescale_intensity(image,
                                    in_range=tuple(percentiles))
    #print(scaled)

    percentile_data = im.fromarray(scaled)
    percentile_data.save(output_name)
    return output_name


