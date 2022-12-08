import numpy as np
import cv2 as cv
import math
from skimage.measure import label, regionprops, EllipseModel
from skimage.measure._regionprops import RegionProperties
from scipy.stats import skew, kurtosis
from scipy import ndimage as ndi
import matplotlib.pyplot as plt


class Blob(RegionProperties):
    """
    A class to get attributes of Blobs

    Description:
    Inherits from the RegionProperties class in scikit-image.
    List of properties that can be obtained are found in
    regionprops documentation:
    "https://scikit-image.org/docs/stable/api/
    skimage.measure.html#skimage.measure.regionprops"

    Attributes
    ----------
    contour : nested list
        contains contour points of blob
        scikit format of contours

    cv_contour : nested list
        same as contour, but with points in double brackets
        opencv format of contours

    orig_image : image matrix
        original image that contour was made from

    label_im : matrix
        labeled image by regions/contours

    Methods (* indicates from RegionProperties)
    -------------------------------------------
    aspect_ratio
    *area
    *area_bbox
    *area_convex
    *area_filled
    *axis_major_length
    *axis_minor_length
    *bbox
    *centroid
    centroid_xy
    *centroid_local
    *centroid_weighted
    *centroid_weighted_local
    circularity
    *coords
    curvature
    curvature_mean
    *eccentricity
    ellipse_fit_residual
    ellipse_fit_residual_mean
    *equivalent_diameter_area
    *euler_number
    *extent
    *feret_diamter_max
    *image
    *image_convex
    image_convex_bbox
    *image_filled
    *image_intensity
    image_mask
    image_mask_bbox
    image_masked
    *inertia_tensor
    *inertia_tensor_eigvals
    *intensity_max
    *intensity_mean
    *intensity_min
    *label
    *moments
    *moments_central
    *moments_hu
    *moments_normalized
    *moments_weighted
    *moments_weighted_central
    *moments_weighted_hu
    *moments_weighted_normalized
    *orientation
    *perimeter
    perimeter_convex_hull
    *perimeter_crofton
    pixel_intensities
    pixel_intensity_mean
    pixel_intensity_median
    pixel_intensity_percentile
    pixel_intensity_std
    pixel_kurtosis
    pixel_skew
    roughness_perimeter
    roughness_surface
    roundness
    *slice
    *solidity
    """

    def __init__(self, contour, orig_image):

        if not isinstance(contour, (list, np.ndarray)):
            raise TypeError('contour must be list or numpy array')

        if not isinstance(orig_image, np.ndarray):
            raise TypeError('orig_image must be a numpy array')

        if orig_image.dtype is not np.uint8:
            orig_image = orig_image.astype('uint8')

        contour = np.array(contour)

        # if opencv contour, remove double brackets
        if len(contour.shape) == 3 and contour.shape[1] == 1:
            self.cv_contour = contour.copy()
            contour = contour[:, 0]
        elif len(contour.shape) == 2:
            self.cv_contour = np.array([[pt] for pt in contour])

        self.contour = contour
        self.orig_image = orig_image

        self.label_im = label(self.image_mask)

        sl = ndi.find_objects(self.label_im)
        super().__init__(slice=sl[0],  # take only slice from list
                         label=1,  # only one slice
                         label_image=self.label_im,
                         intensity_image=orig_image,
                         cache_active=False)

    # ALWAYS USE area_filled and perimeter_crofton for accurate results

    @property
    def aspect_ratio(self):
        """Return aspect ratio as float"""
        return self.axis_major_length / self.axis_minor_length

    @property
    def centroid_xy(self):
        """Return centroid as tuple of numpy.float64"""
        centroid = self.centroid
        # return x,y instead of default y,x
        return centroid[1], centroid[0]

    @property
    def circularity(self):
        """Return circularity as numpy.float64"""
        area = self.area_filled
        perimeter = self.perimeter_crofton
        circularity = (4 * math.pi * area) / pow(perimeter, 2)
        # cannot have circularity above 1 (rounding errors can cause this)
        return min(circularity, 1)

    def curvature(self, num_space=5):
        """
        Return curvature at each contour point as numpy.ndarray

        Parameters
        ----------
        num_space : list of floats
            how far left/right points will be for calculating gradient
        """

        def gradient_spaced(L, num):
            """
            Returns gradient calculated at each point in list

            Parameters
            ----------
            L : list of floats
                gradient will be calculated on these points

            num : int
                how far left/right points will be for calculating gradient

            Returns
            -------
            grad : numpy.ndarray
                array of gradients calculated at each point
            """
            grad = np.array([(L[i+num] - L[i-num])/(num*2)
                             for i in range(-num, len(L)-num)])
            # reorder matrix to align with contour indices
            grad = np.append(grad[num:], grad[0:num])
            return grad

        contour = self.contour

        # curvature can't be calculated
        if len(contour) < 4:
            return np.zeros(len(contour))

        # reduce num_space if contour small enough
        if len(contour) < 50:
            num_space = 1

        dx_dt = gradient_spaced(contour[:, 0], num_space)
        dy_dt = gradient_spaced(contour[:, 1], num_space)

        # velocity
        vel = np.array([[dx_dt[i], dy_dt[i]] for i in range(dx_dt.size)])

        # speed
        ds_dt = np.sqrt(dx_dt * dx_dt + dy_dt * dy_dt)

        # unit tangent vector
        tangent = np.array([1/ds_dt] * 2).transpose() * vel

        d2s_dt2 = gradient_spaced(ds_dt, num_space)
        d2x_dt2 = gradient_spaced(dx_dt, num_space)
        d2y_dt2 = gradient_spaced(dy_dt, num_space)

        curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / \
                          (dx_dt * dx_dt + dy_dt * dy_dt)**1.5

        return curvature

    def curvature_mean(self, num=5):
        """Return mean of curvature as numpy.float64"""
        return np.mean(self.curvature(num))

    @property
    def ellipse_fit_residual(self):
        """
        Returns residual of ellipse fit at each contour point as numpy.ndarray

        Fits the contour to an ellipse, then calculates
        residuals (shortest distance of contour point to
        ellipse fit model)

        This estimates how well the contour fits an ellipse model
        """

        contour = self.contour
        ellipse = EllipseModel()

        # Estimate needed to get params for ellipse, returns True if succeeds
        if ellipse.estimate(contour):
            residuals = ellipse.residuals(contour)
            return residuals
        else:
            return None

    @property
    def ellipse_fit_residual_mean(self):
        """Return mean of residuals as numpy.float64"""
        efr = self.ellipse_fit_residual
        if efr is not None:
            return np.mean(efr)
        else:
            return math.inf

    @property
    def image_convex_bbox(self):
        """Return convex image bbox as numpy.ndarray"""
        im = self.image_convex.astype(np.uint8)
        im[im == 1] = 255
        return im

    @property
    def image_mask(self):
        """Return image with contour as mask as numpy.ndarray"""
        mask = np.zeros(self.orig_image.shape[0:2], np.uint8)
        cv.fillPoly(mask, pts=[self.contour], color=(255, 255, 255))
        return mask

    @property
    def image_mask_bbox(self):
        """Return image with contour as mask in bbox as numpy.ndarray"""
        im = self.image.astype(np.uint8)
        im[im == 1] = 255
        return im

    @property
    def image_masked(self):
        """Return original image with contour as mask as numpy.ndarray"""
        return cv.bitwise_and(self.orig_image,
                              self.orig_image,
                              mask=self.image_mask)

    @property
    def perimeter_convex_hull(self):
        """Return perimeter of convex hull as numpy.float64"""
        convex_label = label(self.image_convex)
        convex_perimeter = regionprops(convex_label)[0]['perimeter_crofton']
        return max(convex_perimeter, 1)

    @property
    def pixel_intensities(self):
        """Return grayscale pixel intensites of
            area inside contour as numpy.ndarray"""
        coords = np.where(self.image_mask == 255)
        if self.orig_image.ndim > 2:
            image_gray = cv.cvtColor(self.image_masked, cv.COLOR_BGR2GRAY)
            return image_gray[coords]
        return self.orig_image[coords]

    @property
    def pixel_intensity_mean(self):
        """Return mean of pixel intensities as numpy.float64"""
        return np.mean(self.pixel_intensities)

    @property
    def pixel_intensity_median(self):
        """Return median of pixel intensities as numpy.float64"""
        return np.median(self.pixel_intensities)

    def pixel_intensity_percentile(self, percentile=75):
        """Return pixel intensity of specified percentile as numpy.uint8"""
        pixel_sort = np.sort(self.pixel_intensities)
        idx = int(percentile/100*len(pixel_sort))
        return pixel_sort[idx]

    @property
    def pixel_intensity_std(self):
        """Return standard deviation of pixel intensities as numpy.float64"""
        return np.std(self.pixel_intensities)

    @property
    def pixel_kurtosis(self):
        """Return kurtosis of pixel intensities as float"""
        return kurtosis(self.pixel_intensities, fisher=True, bias=False)

    @property
    def pixel_skew(self):
        """Return skew of pixel intensities as float"""
        return skew(self.pixel_intensities, bias=False, nan_policy='omit')

    @property
    def roughness_perimeter(self):
        """Return perimeter roughness as numpy.float64"""
        roughness = self.perimeter_crofton / self.perimeter_convex_hull
        # perimeter roughness cannot be less than 1 (rounding errors)
        return max(roughness, 1)

    @property
    def roughness_surface(self):
        """Return surface roughness as numpy.float64"""
        pixels = self.pixel_intensities
        mean = self.pixel_intensity_mean
        diff = [abs(px-mean) for px in pixels]
        return sum(diff)/len(diff)

    @property
    def roundness(self):
        """Return roundness as numpy.float64"""
        num = 4 * self.area_filled
        den = math.pi * pow(self.axis_major_length, 2)
        return num / den

    def print_properties(self, dec=2):
        funcs = [
            'aspect_ratio',
            'area_filled',
            'area_convex',
            'axis_major_length',
            'axis_minor_length',
            'centroid_xy',
            'circularity',
            'curvature_mean',
            'eccentricity',
            'ellipse_fit_residual_mean',
            'equivalent_diameter_area',
            'orientation',
            'perimeter_crofton',
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
            try:
                val = eval('self.' + funcs[i])
                print(funcs[i] + ': ' + str(np.around(val, dec)))
            except Exception as e:
                val = eval('self.' + funcs[i] + '()')
                print(funcs[i] + ': ' + str(np.around(val, dec)))


def plot_image(blob):
    x0, y0 = blob.centroid_xy
    y0 = int(y0)
    x0 = int(x0)
    orientation = blob.orientation
    x1 = int(x0 + math.cos(orientation) * 0.5 * blob.axis_minor_length)
    y1 = int(y0 - math.sin(orientation) * 0.5 * blob.axis_minor_length)
    x2 = int(x0 - math.sin(orientation) * 0.5 * blob.axis_major_length)
    y2 = int(y0 - math.cos(orientation) * 0.5 * blob.axis_major_length)
    
    im = blob.orig_image
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
    ret, im_thresh = cv.threshold(im_blur, 125, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(im_thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv.contourArea)
    blob = Blob(contour, im)
    print(blob.area_convex)
    blob.print_properties()
    #plot_image(blob)
    '''
    cv.imshow('masked', blob.image_masked)
    plt.hist(blob.pixel_intensities,256,[0,256]); plt.show()
    cv.waitKey()
    '''


if __name__ == '__main__':
    main()
