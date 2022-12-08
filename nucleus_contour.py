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
    num_iter: number of iterations for snake
    smoothing: amount of smoothing for snake

    Returns:
    --------
    contour: the contour given by the snake

    '''
    img = np.array(image, dtype=np.uint8)
    
    # convert image to grayscale if needed
    if len(image.shape) > 3:
        img = rgb2gray(img)
    
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

    snake = snake.astype(np.uint8)

    contours, _ = cv.findContours(snake,
                                  cv.RETR_EXTERNAL,
                                  cv.CHAIN_APPROX_NONE)

    contour = max(contours, key=cv.contourArea)

    return contour


if __name__ == '__main__':
    image = cv.imread('ex3.tif')
    contour = nucleus_contour(image)
    Nuc_blob = bc.Blob(contour, image)
    print(Nuc_blob.area_filled)
    print(Nuc_blob.perimeter_crofton)
    
    cv.drawContours(image, contour, -1, (255,0,0), 2, cv.LINE_8)
    cv.imshow('img', image)
    cv.waitKey()