import numpy as np
import cv2 as cv
import math
from skimage.measure import label, regionprops
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt


class Blob():

    def __init__(self, image, contour):
        self.image = image
        self.contour = contour
        self.mask = np.zeros(self.image.shape[0:2], np.uint8)
        cv.fillPoly(self.mask, pts=[contour], color=(255, 255, 255))
        self.label_img = label(self.mask)
        self.region = regionprops(self.label_img)[0]
    
    @property
    def aspect_ratio(self):
        return self.axis_major_length / self.axis_minor_length

    @property
    def area(self):
        return self.region.area_filled

    @property
    def area_convex_hull(self):
        return self.region.area_convex

    @property
    def axis_major_length(self):
        return self.region.axis_major_length

    @property
    def axis_minor_length(self):
        return self.region.axis_minor_length

    @property
    def centroid(self):
        return self.region.centroid

    @property
    def circularity(self):
        area = self.area
        perimeter = self.perimeter
        circularity = (4 * math.pi * area) / pow(perimeter, 2)
        return circularity
        
    @property
    def eccentricity(self):
        return self.region.eccentricity

    @property
    def equivalent_diameter_area(self):
        return self.region.equivalent_diameter_area
    
    @property
    def image_convex_bbox(self):
        im = self.region.image_convex.astype(np.uint8)
        im[im == 1] = 255
        return im

    @property
    def image_mask_bbox(self):
        im = self.region.image.astype(np.uint8)
        im[im == 1] = 255
        return im

    @property
    def image_masked(self):
        '''Original image with mask'''
        return cv.bitwise_and(self.image,
                              self.image,
                              mask=self.mask)

    @property
    def orientation(self):
        return self.region.orientation
    
    @property
    def perimeter(self):
        return self.region.perimeter
    
    @property
    def perimeter_convex_hull(self):
        convex_label = label(self.image_convex_bbox)
        convex_perimeter = regionprops(convex_label)[0]['perimeter']
        return convex_perimeter
    
    @property
    def pixel_intensities(self):
        coords = np.where(self.mask == 255)
        if len(self.image.shape) >= 3:
            image_gray = cv.cvtColor(self.image_masked, cv.COLOR_BGR2GRAY)
            return image_gray[coords]
        return self.image[coords]
    
    @property
    def pixel_intensity_mean(self):
        return np.mean(self.pixel_intensities)
    
    @property
    def pixel_intensity_median(self):
        return np.median(self.pixel_intensities)
    
    @property
    def pixel_kurtosis(self):
        return kurtosis(self.pixel_intensities, fisher=True, bias=False)
    
    @property
    def pixel_skew(self):
        return skew(self.pixel_intensities, bias=False, nan_policy='omit')
            
    @property
    def roughness(self):
        return self.perimeter / self.perimeter_convex_hull

    @property
    def roundness(self):
        num = 4 * self.area
        den = math.pi * pow(self.axis_major_length, 2)
        return num / den

    @property
    def solidity(self):
        return self.region.solidity
    
    def pixel_intensity_at_percent(self, percent=75):
        pixel_sort = np.sort(self.pixel_intensities)
        idx = int(percent/100*len(pixel_sort))
        return pixel_sort[idx]
    
    def print_properties(self, dec=2):
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
            'pixel_intensity_mean',
            'pixel_intensity_median',
            'pixel_kurtosis',
            'pixel_skew',
            'roughness',
            'roundness',
            'solidity'
            ]

        for i in range(len(funcs)):
            val = eval('self.' + funcs[i])
            print(funcs[i] + ': ' + str(np.around(val, dec)))

def plot_image(blob):
    y0, x0 = blob.centroid
    y0 = int(y0)
    x0 = int(x0)
    orientation = blob.orientation
    x1 = int(x0 + math.cos(orientation) * 0.5 * blob.axis_minor_length)
    y1 = int(y0 - math.sin(orientation) * 0.5 * blob.axis_minor_length)
    x2 = int(x0 - math.sin(orientation) * 0.5 * blob.axis_major_length)
    y2 = int(y0 - math.cos(orientation) * 0.5 * blob.axis_major_length)
    
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
    cv.imshow('masked', blob.image_masked)
    cv.waitKey()
    
def main():
    im = cv.imread("ex3.tif")
    im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    im_blur = cv.GaussianBlur(im_gray, (25, 25), 0)
    ret, im_thresh = cv.threshold(im_blur, 125, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(im_thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv.contourArea)
    blob = Blob(im, contour)
    blob.print_properties(2)
    print(blob.pixel_intensity_at_percent(75))
    #plot_image(blob)
    #cv.imshow('gray', blob.image_gray)
    #cv.imshow('orig', blob.image_original_masked())
    #cv.waitKey()
    
    cv.imshow('masked', blob.image_masked)
    plt.hist(blob.pixel_intensities,256,[0,256]); plt.show()
    cv.waitKey()
    


if __name__ == '__main__':
    main()