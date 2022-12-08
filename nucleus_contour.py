from skimage.color import rgb2gray
from skimage import io
from skimage.segmentation import (checkerboard_level_set,
                                  morphological_chan_vese)
from skimage.morphology import disk
import numpy as np
import matplotlib.pyplot as plt
import contrast_functions as cfs
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


def nucleus_contour(image, num_iter=70, smoothing=3):
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
    img = rgb2gray(image)

    # Initial level set

    init_ls = checkerboard_level_set(np.shape(img), 5)

    # List with intermediate results for plotting the evolution

    evolution = []
    callback = store_evolution_in(evolution)

    # Chan_vese

    snake = morphological_chan_vese(image=img,
                                    num_iter=num_iter,
                                    init_level_set=init_ls,
                                    smoothing=smoothing,
                                    iter_callback=callback)

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.imshow(img, cmap="gray")
    ax.set_axis_off()
    ax.contour(snake, [.5], colors='r')

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
    xs = vertices[:, 0]
    ys = vertices[:, 1]
    contour = [[int(xs[i]), int(ys[i])] for i in range(len(xs)-1)]

    return contour


if __name__ == '__main__':
    image = cv.imread('ex3.tif')
    contour = nucleus_contour(image)
    Nuc_blob = bc.Blob(contour, image)
    print(Nuc_blob.area_filled)
    print(Nuc_blob.perimeter_crofton)

    contour = np.array(contour)
    print(contour.shape)
    
    cv.drawContours(image, Nuc_blob.cv_contour, -1, (255,0,0), 2, cv.LINE_8)
    cv.imshow('img', image)
    cv.waitKey()