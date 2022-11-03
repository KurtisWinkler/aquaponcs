import numpy as np
import matplotlib.pyplot as plt
from skimage import io, exposure, data
from PIL import Image as im

# contour detection documentation
# https://danielmuellerkomorowska.com/2020/06/27/contrast-adjustment-with-scikit-image/

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, exposure, data
from PIL import Image as im
 
# read the image
def contrast(image_name, output_name):
    image = io.imread(image_name)

def main():
    min_max_rescale('ex1.tif', 'minmax_rescaled_ex1.png')
    percentile_rescale('ex1.tif', 0.5, 99.5, 'percentile_rescaled_ex1.png')

def min_max_rescale(file, output_name):
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
    # Read the image 
    try: 
        image = io.imread(file)
    except: 
        FileNotFoundError
        sys.exit(1)

    # Reports pixel intensity values on a scale of 0 to 255 for an 8-bit image
    image.max()
    image.min()

    # Rescale the min and max pixel intensity values

    image_minmax_scaled = exposure.rescale_intensity(image)
    image_minmax_scaled.max()
    # 255
    image_minmax_scaled.min()
    # 0

    minmax_data = im.fromarray(image_minmax_scaled)
    minmax_data.save(output_name)
    
    return minmax_data


def percentile_rescale(file, min_percentile, max_percentile, output_name):
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
    output_name: str
        name of file path for the output image to be saved as .png
        
    Output: 
    minmax_data: array 
        numpy array of pixel intensities to be saved as a .png
    '''
    # Read the image 
    try: 
        image = io.imread(file)
    except: 
        FileNotFoundError
        sys.exit(1)
    
    # Rescale pixel intensity by percentile
    percentiles = np.percentile(image, (min_percentile, max_percentile))
    scaled = exposure.rescale_intensity(image,
                                        in_range=tuple(percentiles))

    percentile_data = im.fromarray(scaled)
    percentile_data.save(output_name)
    
    return percentile_data


    
if __name__ == '__main__':
    main()