'''
This code uses the active_contours function from scikit-image to create a snake that fits to the nucleus of the cell
'''
from skimage.color import rgb2gray
from skimage import io
from skimage.segmentation import (checkerboard_level_set, morphological_chan_vese)
from skimage.morphology import disk
import numpy as np
import matplotlib.pyplot as plt
import contrast_function_scikit as cfs
import blob_class as bc
import cv2 as cv

input_name = 'test_image.jpeg'
output_name_contrast = 'contrasted_image.jpeg'
output_name = 'unit_test.jpeg'

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
    image: a matrix of the desired image
    input_name: a str name of an input image file
    output_name_contrast: a str name of the contrasted output image
    output_name: a str name of the final output image with contour
    
    Returns:
    --------
    output_name: a str name of the final image with the contour drawn
    
    '''

    contrast_image = cfs.percentile_rescale(input_name, 0.5, 99.5, output_name_contrast)
    image_cont = io.imread(output_name_contrast)
    img = rgb2gray(image_cont)

    # Initial level set
    
    init_ls = checkerboard_level_set(np.shape(img), 5)
    
    # List with intermediate results for plotting the evolution
    
    evolution = []
    callback = store_evolution_in(evolution)
    
    snake = morphological_chan_vese(img, num_iter=70, init_level_set=init_ls,
                             smoothing=3, iter_callback=callback)

    fig, ax = plt.subplots(figsize=(7, 7))
    
    ax.imshow(img, cmap="gray")
    ax.set_axis_off()
    ax.contour(snake, [.5], colors='r')
    
    plot = plt.savefig(output_name, bbox_inches='tight')
    
    contours = ax.contour(snake, [.5])
    
    # Finding the longest path to avoid small contours on image noise
    
    paths = []
    for path in contours.collections[0].get_paths():
        paths.append(path)
        
    lengths = []
    for path in paths:
        lengths.append(len(path))
        
    max_length = max(lengths)
    
    path_idx = lengths.index(max_length)
    
    cell_path = paths[path_idx]
    vertices = cell_path.vertices
    xs = vertices[:,0]
    ys = vertices[:,1]
    contour = [(int(xs[i]),int(ys[i])) for i in range(len(xs)-1)]
    
    return output_name, contour, img

if __name__ == '__main__':
    output_name, contour, img = nucleus_contour(input_name, output_name_contrast, output_name)
    Nuc_blob = bc.Blob(contour, img)
    print(Nuc_blob.area_filled)
    print(Nuc_blob.perimeter_crofton)
