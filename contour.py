''' from https://docs.opencv.org/3.4/df/d0d/tutorial_find_contours.html '''
'''
from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
import random as rng
rng.seed(12345)

def thresh_callback(val):
    threshold = val
    
    # Detect edges using Canny
    canny_output = cv.Canny(src_gray, threshold, threshold * 2)
    
    # Find contours
    contours, hierarchy = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    # Draw contours
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    
    for i in range(len(contours)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv.drawContours(drawing, contours, i, color, 2, cv.LINE_8, hierarchy, 0)
        
    # Show in a window
    cv.imshow('Contours', drawing)
    
# Load source image
parser = argparse.ArgumentParser(description='Code for Finding contours in your image tutorial.')
parser.add_argument('--input', help='Path to input image.', default='HappyFish.jpg')
args = parser.parse_args()
src = cv.imread(cv.samples.findFile(args.input))

if src is None:
    print('Could not open or find the image:', args.input)
    exit(0)
    
# Convert image to gray and blur it
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
src_gray = cv.blur(src_gray, (3,3))

# Create Window
source_window = 'Source'
cv.namedWindow(source_window)
cv.imshow(source_window, src)
max_thresh = 255
thresh = 100 # initial threshold
cv.createTrackbar('Canny Thresh:', source_window, thresh, max_thresh, thresh_callback)
thresh_callback(thresh)
cv.waitKey()
'''

import numpy as np
import cv2 as cv
im = cv.imread('Equalized_image.png')
imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
imgray = cv.blur(imgray, (15,15))
ret, thresh = cv.threshold(imgray, 51, 255, cv.THRESH_BINARY)
cv.imwrite('blur_gray.png', imgray)
cv.imwrite('thresh.png', thresh)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(thresh, contours, -1, (0,255,0), 3)
cv.imwrite('source.png', im)
cv.imwrite('Contours.png', thresh)
