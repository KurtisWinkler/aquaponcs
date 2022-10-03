'''from https://docs.opencv.org/3.4/d4/d1b/tutorial_histogram_equalization.html '''

import cv2 as cv
import argparse

# Load source image
im = cv.imread("ex1.tif")
if im is None:
    print('Could not open or find the image:', args.input)
    exit(0)

# Gray and Blur image
im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

dst = cv.equalizeHist(im_gray)

clahe = cv.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
dst2 = clahe.apply(im_gray)
#dst2 = cv.equalizeHist(dst2)

cv.imshow('Source_image', im)
cv.imshow('Equal_Image.png', dst)
cv.imshow('CLAHE_image', dst2)


cv.waitKey()
