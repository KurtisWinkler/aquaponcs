'''Set of functions to describe blobs/circles'''
import cv2 as cv
import math

def get_center(contour):
    M = cv.moments(contour)
    x = int(M['m10']/M['m00'])
    y = int(M['m01']/M['m00'])
    return x, y

def get_circularity(contour):
    M = cv.moments(contour)
    area = M['m00']
    perimeter = cv.arcLength(contour, True)
    circularity = (4 * math.pi * area) / pow(perimeter, 2)
    return circularity
    