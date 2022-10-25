import numpy as np
import cv2 as cv
import math
from skimage.measure import label, regionprops, EllipseModel
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt


class Blob():

    def __init__(self, contour, image):
        contour = np.array(contour)
        
        # if opencv contour, remove double brackets
        if len(contour.shape) == 3 and contour.shape[1] == 1:
            self.cv_contour = contour.copy()
            contour = contour[:,0]

        self.contour = contour
        self.image = image

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
        centroid = self.region.centroid
        # return x,y instead of default y,x
        return centroid[1], centroid[0]

    @property
    def circularity(self):
        area = self.area
        perimeter = self.perimeter
        circularity = (4 * math.pi * area) / pow(perimeter, 2)
        # cannot have circularity above 1 (rounding errors can cause this)
        return min(circularity, 1)

    @property
    def coords(self):
        points = np.where(self.image_mask == 255)
        coords = [[points[0][i], points[1][i]] for i in range(len(points[0]))]
        return coords
    
    @property
    def curvature(self, num=5):
        
        def gradient_spaced(L,num):
            grad = np.array([(L[i+num] - L[i-num])/(num*2) for i in range(-num,len(L)-num)])
            # reorder matrix to align with contour indices
            grad = np.append(grad[num:], grad[0:num])
            return grad

        contour = self.contour
        
        dx_dt = gradient_spaced(contour[:, 0], num)
        dy_dt = gradient_spaced(contour[:, 1], num)

        # velocity
        vel = np.array([[dx_dt[i], dy_dt[i]] for i in range(dx_dt.size)])

        # speed
        ds_dt = np.sqrt(dx_dt * dx_dt + dy_dt * dy_dt)

        # unit tangent vector
        tangent = np.array([1/ds_dt] * 2).transpose() * vel

        d2s_dt2 = gradient_spaced(ds_dt, num)
        d2x_dt2 = gradient_spaced(dx_dt, num)
        d2y_dt2 = gradient_spaced(dy_dt, num)

        curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt)**1.5
        
        return curvature
    
    @property
    def eccentricity(self):
        return self.region.eccentricity

    @property
    def ellipse_fit_mean_residual(self):
        ''' Fits the contour to an ellipse, then returns the mean
            residuals (shortest distance of contour point to 
            ellipse model)
            
            This estimates how well the contour fits an ellipse model
            '''
        
        contour = self.contour
        ellipse = EllipseModel()

        # Estimate needed to get params for ellipse, returns True if succeeds
        if ellipse.estimate(contour):
            residuals = ellipse.residuals(contour)
            return np.mean(residuals)
        else:
            return math.inf
    
    @property
    def equivalent_diameter_area(self):
        return self.region.equivalent_diameter_area

    @property
    def image_convex_bbox(self):
        im = self.region.image_convex.astype(np.uint8)
        im[im == 1] = 255
        return im

    @property
    def image_mask(self):
        mask = np.zeros(self.image.shape[0:2], np.uint8)
        cv.fillPoly(mask, pts=[self.contour], color=(255, 255, 255))
        return mask

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
                              mask=self.image_mask)

    @property
    def moments(self):
        return self.region.moments
    
    @property
    def orientation(self):
        return self.region.orientation

    @property
    def perimeter(self):
        return max(self.region.perimeter_crofton, 1) #won't equal 0

    @property
    def perimeter_convex_hull(self):
        convex_label = label(self.image_convex_bbox)
        convex_perimeter = regionprops(convex_label)[0]['perimeter_crofton']
        return max(convex_perimeter, 1)

    @property
    def pixel_intensities(self):
        coords = np.where(self.image_mask == 255)
        if self.image.ndim > 2:
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
    def pixel_intensity_std(self):
        return np.std(self.pixel_intensities)

    @property
    def pixel_kurtosis(self):
        return kurtosis(self.pixel_intensities, fisher=True, bias=False)

    @property
    def pixel_skew(self):
        return skew(self.pixel_intensities, bias=False, nan_policy='omit')

    @property
    def region(self):
        '''blob inherits regionprops properties from scikit class'''
        label_img = label(self.image_mask)
        region = regionprops(label_img)[0]
        return region

    @property
    def residual_corrected(self):
        return self.ellipse_fit_mean_residual / self.perimeter
    
    @property
    def roughness_perimeter(self):
        roughness = self.perimeter / self.perimeter_convex_hull
        # perimeter roughness cannot be less than 1 (rounding errors)
        return max(roughness, 1)
    
    @property
    def roughness_surface(self):
        pixels = self.pixel_intensities
        mean = self.pixel_intensity_mean
        diff = [abs(px-mean) for px in pixels]
        return sum(diff)/len(diff)

    @property
    def roundness(self):
        num = 4 * self.area
        den = math.pi * pow(self.axis_major_length, 2)
        return num / den

    @property
    def solidity(self):
        return self.region.solidity

    def pixel_intensity_percentile(self, percentile=75):
        pixel_sort = np.sort(self.pixel_intensities)
        idx = int(percentile/100*len(pixel_sort))
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
            'ellipse_fit_mean_residual',
            'equivalent_diameter_area',
            'orientation',
            'perimeter',
            'perimeter_convex_hull',
            'pixel_intensity_mean',
            'pixel_intensity_median',
            'pixel_intensity_std',
            'pixel_kurtosis',
            'pixel_skew',
            'roughness_perimeter',
            'roughness_surface',
            'roundness',
            'solidity'
            ]

        for i in range(len(funcs)):
            val = eval('self.' + funcs[i])
            print(funcs[i] + ': ' + str(np.around(val, dec)))


def plot_image(blob):
    x0, y0 = blob.centroid
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
    cv.drawContours(im_copy, blob.cv_contour, -1, (255,0,0), 2, cv.LINE_8)
    cv.imshow('orig', im)
    cv.imshow('params', im_copy)
    cv.imwrite('pres1.jpg', im_copy)
    cv.imshow('masked', blob.image_masked)
    cv.waitKey()
    
def main():
    im = cv.imread("ex3.tif")
    im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    im_blur = cv.GaussianBlur(im_gray, (25, 25), 0)
    ret, im_thresh = cv.threshold(im_blur, 25, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(im_thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv.contourArea)
    blob = Blob(contour, im)
    blob.print_properties(2)
    #plot_image(blob)
    #cv.imshow('gray', blob.image_gray)
    #cv.imshow('orig', blob.image_original_masked())
    #cv.waitKey()
    '''
    cv.imshow('masked', blob.image_masked)
    plt.hist(blob.pixel_intensities,256,[0,256]); plt.show()
    cv.waitKey()
    '''
    
if __name__ == '__main__':
    main()