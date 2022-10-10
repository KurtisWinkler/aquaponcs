import numpy as np
import cv2 as cv
import math
from skimage.measure import label, regionprops


class Blob():

    def __init__(self, image, contour):
        self.image = image
        self.contour = contour
        self.mask = np.zeros(self.image.shape[0:2], np.uint8)
        cv.fillPoly(self.mask, pts=[contour], color=(255, 255, 255))
        self.label_img = label(self.mask)
        self.region = regionprops(self.label_img)[0]

    def aspect_ratio(self):
        return self.axis_major_length() / self.axis_minor_length()

    def area(self):
        return self.region.area_filled

    def area_convex_hull(self):
        return self.region.area_convex

    def axis_major_length(self):
        return self.region.axis_major_length

    def axis_minor_length(self):
        return self.region.axis_minor_length

    def centroid(self):
        return self.region.centroid

    def circularity(self):
        area = self.area()
        perimeter = self.perimeter()
        circularity = (4 * math.pi * area) / pow(perimeter, 2)
        return circularity

    def eccentricity(self):
        return self.region.eccentricity

    def equivalent_diameter_area(self):
        return self.region.equivalent_diameter_area
    
    def image_convex(self):
        im = self.region.image_convex.astype(np.uint8)
        im[im == 1] = 255
        return im

    def image_mask(self):
        im = self.region.image.astype(np.uint8)
        im[im == 1] = 255
        return im

    def image_original_masked(self):
        return cv.bitwise_and(self.image,
                              self.image,
                              mask=self.mask)

    def orientation(self):
        return self.region.orientation
    
    def perimeter(self):
        return self.region.perimeter
    
    def perimeter_convex_hull(self):
        convex_label = label(self.image_convex())
        convex_perimeter = regionprops(convex_label)[0]['perimeter']
        return convex_perimeter
    
    def roughness(self):
        return self.perimeter() / self.perimeter_convex_hull()

    def roundness(self):
        num = 4 * self.area()
        den = math.pi * pow(self.axis_major_length(), 2)
        return num / den

    def solidity(self):
        return self.region.solidity

    def zproperties(self, dec=2):
        funcs = [
            'aspect_ratio',
            'area',
            'area_convex_hull',
            'axis_major_length',
            'axis_minor_length',
            'centroid',
            'circularity',
            'eccentricity',
            'equivalent_diameter_area',
            'orientation',
            'perimeter',
            'perimeter_convex_hull',
            'roughness',
            'roundness',
            'solidity',
            ]

        for i in range(len(funcs)):
            val = eval('self.' + funcs[i] + '()')
            print(funcs[i] + ': ' + str(np.around(val, dec)))

def plot_image(blob):
    y0, x0 = blob.centroid()
    y0 = int(y0)
    x0 = int(x0)
    orientation = blob.orientation()
    x1 = int(x0 + math.cos(orientation) * 0.5 * blob.axis_minor_length())
    y1 = int(y0 - math.sin(orientation) * 0.5 * blob.axis_minor_length())
    x2 = int(x0 - math.sin(orientation) * 0.5 * blob.axis_major_length())
    y2 = int(y0 - math.cos(orientation) * 0.5 * blob.axis_major_length())
    
    im = blob.image
    im_copy = im.copy()
    
    #cv.line(im_copy, (x0, y0), (x1, y1), (0,255,255), 2)
    cv.line(im_copy, (x0, y0), (x0-(x1-x0), y0-(y1-y0)), (0,255,255), 2)
    #cv.line(im_copy, (x0, y0), (x2, y2), (0,0,255), 2)
    cv.line(im_copy, (x0, y0), (x0-(x2-x0), y0-(y2-y0)), (0,0,255), 2)
    cv.circle(im_copy, (x0, y0), 2, (0,255,0), 2)
    cv.drawContours(im_copy, blob.contour, -1, (255,0,0), 2, cv.LINE_8)
    
    cv.imshow('orig', im)
    cv.imshow('params', im_copy)
    cv.imshow('masked', blob.image_original_masked())
    cv.waitKey()
    

def main():
    im = cv.imread("ex3.tif")
    im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    im_blur = cv.GaussianBlur(im_gray, (25, 25), 0)
    ret, im_thresh = cv.threshold(im_blur, 25, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(im_thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv.contourArea)
    blob = Blob(im, contour)
    blob.zproperties(2)
    plot_image(blob)
    #cv.imshow('gray', blob.image_gray)
    #cv.imshow('orig', blob.image_original_masked())
    #cv.waitKey()


if __name__ == '__main__':
    main()