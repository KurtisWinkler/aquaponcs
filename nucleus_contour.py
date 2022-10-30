'''
This code uses the active_contours function from scikit-image to create a snake that fits to the nucleus of the cell
'''
from skimage.draw import circle_perimeter
from skimage.filters import gaussian, sobel
from skimage.filters.rank import entropy
from skimage.color import rgb2gray
from skimage import io
from skimage.segmentation import (active_contour, inverse_gaussian_gradient, morphological_geodesic_active_contour, disk_level_set, checkerboard_level_set, morphological_chan_vese)
from skimage.morphology import disk
import numpy as np
import matplotlib.pyplot as plt
import contrast_function_scikit as cfs

input_name = 'test_1.jpeg'
output_name_contrast = 'contrasted_image.png'
output_name = 'new_test.png'

def store_evolution_in(L):
    """Returns a callback function to store the evolution of the level sets in
    the given list.
    """

    def store(x):
        L.append(np.copy(x))

    return store

def nucleus_contour(input_name, output_name_contrast, output_name):
    '''
    Inputs:
    -------
    input_name: a str name of an input image file
    output_name_contrast: a str name of the contrasted output image
    output_name: a str name of the final output image with contour
    
    Returns:
    --------
    output_name: a str name of the final image with the contour drawn
    
    '''
    
    contrast_image = cfs.contrast(input_name, output_name_contrast)

    img = io.imread(contrast_image)
    img = rgb2gray(img)

    # Initial level set
    
    init_ls = checkerboard_level_set(np.shape(img),10)
    
    # List with intermediate results for plotting the evolution
    
    evolution = []
    callback = store_evolution_in(evolution)

    ## Chan_vese
    
    ## Entropy
    
    # I think I can get a more accurate perimeter if I mask using the entropy, but it's not currently necessary
    # For the perimeter finding
    
    # ent_fig, ent_ax = plt.subplots(figsize=(7,7))
    # entropy_image = entropy(img, disk(5))
    # ent_ax.imshow(entropy_image, cmap = 'magma')
    # plot = plt.savefig('entropy_image.png', bbox_inches='tight')
    
    snake = morphological_chan_vese(img, num_iter=70, init_level_set=init_ls,
                             smoothing=10, iter_callback=callback)


    fig, ax = plt.subplots(figsize=(7, 7))
    
    ax.imshow(img, cmap="gray")
    ax.set_axis_off()
    ax.contour(snake, [0.5], colors='r')
    
    ax.plot(init_ls[:, 1], init_ls[:, 0], '--r', lw=3)
    ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)

    plot = plt.savefig(output_name, bbox_inches='tight')
    
    return output_name

if __name__ == '__main__':
    nucleus_contour(input_name, output_name_contrast, output_name)
