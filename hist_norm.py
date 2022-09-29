'''from https://docs.opencv.org/3.4/d4/d1b/tutorial_histogram_equalization.html '''

import cv2 as cv
import argparse

parser = argparse.ArgumentParser(description='Code for Histogram Equalization tutorial.')
parser.add_argument('--input', help='Path to input image.', default='lena.jpg')
args = parser.parse_args()
src = cv.imread(cv.samples.findFile(args.input))
if src is None:
    print('Could not open or find the image:', args.input)
    exit(0)
src = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
dst = cv.equalizeHist(src)
cv.imwrite('Source_image.png', src)
cv.imwrite('Equalized_Image.png', dst)
