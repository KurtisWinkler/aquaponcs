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
    print('area: ' + str(area))
    perimeter = cv.arcLength(contour, True)
    circularity = (4 * math.pi * area) / pow(perimeter, 2)
    return circularity

def get_solidity(contour):
    M = cv.moments(contour)
    area = M['m00']
    hull = cv.convexHull(contour)
    hull_area = cv.contourArea(hull)
    solidity = float(area)/hull_area
    return solidity

def get_aspectRatio(contour):
    (x,y),(MA,ma),angle = cv.fitEllipse(contour)
    aspect_ratio = float(ma)/MA
    return aspect_ratio

def get_roundness(contour):
    M = cv.moments(contour)
    area = M['m00']
    (x,y),(MA,ma),angle = cv.fitEllipse(contour)
    roundness = (4 * area)/(math.pi * pow(ma, 2))
    return roundness

def main(): #FOR TESTING
    # CREATE CONTOURS
    im = cv.imread("ex3.tif")
    im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    im_blur = cv.GaussianBlur(im_gray, (25,25), 0)
    ret, im_thresh = cv.threshold(im_blur, 25, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(im_thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv.contourArea)
    
    # TEST CREATED FUNCTIONS
    contour_area = cv.contourArea(contour)
    contour_perimeter = cv.arcLength(contour, True)
    contour_circularity = get_circularity(contour)
    cx, cy = get_center(contour)
    contour_solidity = get_solidity(contour)
    contour_ratio = get_aspectRatio(contour)
    roundness = get_roundness(contour)

    # PRINT VALUES
    print('Contour Area: ' + str(contour_area))
    print('Contour Perimeter: ' + str(round(contour_perimeter,1)))
    print('Contour Circularity: ' + str(round(contour_circularity,3)))
    print('Contour Center: (' + str(cx) + ',' + str(cy) + ')')
    print('Contour Solidity: ' + str(round(contour_solidity,3)))
    print('Contour Aspect Ratio: ' + str(round(contour_ratio,3)))
    print('Contour Roundness: ' + str(round(roundness,3)))
    
    
    # DRAW/DISPLAY IMAGES
    im_copy = im.copy()
    cv.drawContours(im_copy, contours, -1, (0, 255, 0), 2, cv.LINE_8)
    cv.imshow('source_window', im)
    cv.imshow('Contours', im_copy)
    cv.waitKey()

if __name__ == '__main__':
    main()