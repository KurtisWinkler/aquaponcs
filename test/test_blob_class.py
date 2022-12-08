import unittest
import numpy as np
from scipy import ndimage as ndi
from scipy.stats import skew, kurtosis
import math
import cv2 as cv
import os
import sys
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(src_path)
import blob_class as bc


class TestBlobDetection(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        im = np.zeros((512, 512, 3), dtype='uint8')

        # specify color and thickness for each created blob
        color = (255, 255, 255)
        thickness = -1

        # create circle
        center_coordinates = (150, 200)
        radius = 50
        im_c = im.copy()
        cv.circle(im_c, center_coordinates, radius, color, thickness)
        cls.im_dist_c = cv.cvtColor(im_c, cv.COLOR_BGR2GRAY)
        cls.im_dist_c = np.array(ndi.distance_transform_edt(cls.im_dist_c),
                                 dtype='uint8')

        # create ellipse
        center_coordinates = (250, 400)
        axesLength = (100, 50)
        angle = 30
        startAngle = 0
        endAngle = 360
        im_e = im.copy()
        cv.ellipse(im_e, center_coordinates, axesLength, angle,
                   startAngle, endAngle, color, thickness)

        # make blobs
        im_cg = cv.cvtColor(im_c, cv.COLOR_BGR2GRAY)
        im_eg = cv.cvtColor(im_e, cv.COLOR_BGR2GRAY)

        cls.contour_c, _ = cv.findContours(im_cg,
                                           cv.RETR_EXTERNAL,
                                           cv.CHAIN_APPROX_NONE)
        cls.contour_e, _ = cv.findContours(im_eg,
                                           cv.RETR_EXTERNAL,
                                           cv.CHAIN_APPROX_NONE)

        # create blobs for testing
        cls.blob_circle = bc.Blob(cls.contour_c[0], im_c)
        cls.blob_circle_dist = bc.Blob(cls.contour_c[0], cls.im_dist_c)
        cls.blob_ellipse = bc.Blob(cls.contour_e[0], im_e)

    @classmethod
    def tearDownClass(cls):
        cls.blob_cirlce = None
        cls.blob_circle_dist = None
        cls.im_dist_c = None
        cls.blob_ellipse = None
        cls.contour_c = None
        cls.contour_e = None

    def test_error(self):
        # TypeError if contour (input 1) not list or numpy array
        self.assertRaises(TypeError, bc.Blob, 'a', self.im_dist_c)

        # TypeError if orig_image (input 2) not numpy array
        self.assertRaises(TypeError, bc.Blob, self.contour_c, [1, 2, 3])

    def test_aspect_ratio(self):
        # check within one decimal point
        real_c = 1
        circle = self.blob_circle.aspect_ratio
        self.assertAlmostEqual(circle, real_c, places=1)

        real_e = 2
        ellipse = self.blob_ellipse.aspect_ratio
        self.assertAlmostEqual(ellipse, real_e, places=1)

    def test_centroid_xy(self):
        real_c = [150, 200]
        circle = self.blob_circle.centroid_xy
        self.assertAlmostEqual(circle[0], real_c[0], places=1)
        self.assertAlmostEqual(circle[1], real_c[1], places=1)

        real_e = [250, 400]
        ellipse = self.blob_ellipse.centroid_xy
        self.assertAlmostEqual(ellipse[0], real_e[0], places=1)
        self.assertAlmostEqual(ellipse[1], real_e[1], places=1)

    def test_circularity(self):
        # area is pi * 50^2
        # perimeter is pi * 100
        real_c = (4 * math.pi * math.pi * 50 * 50) / pow(100 * math.pi, 2)
        circle = self.blob_circle.circularity
        self.assertAlmostEqual(circle, real_c, places=1)

        # area is pi * 100 * 50
        # perimeter is 484.82 (complicated formula so calculated with google)
        real_e = (4 * math.pi * math.pi * 100 * 50) / pow(484.42, 2)
        ellipse = self.blob_ellipse.circularity
        self.assertAlmostEqual(ellipse, real_e, places=1)

    def test_curvature_mean(self):
        # radius of 50
        real_c = 1 / 50
        circle = self.blob_circle.curvature_mean()
        self.assertAlmostEqual(circle, real_c, places=1)

    def test_ellipse_fit_residual_mean(self):
        real_c = 0
        circle = self.blob_circle.ellipse_fit_residual_mean
        self.assertAlmostEqual(circle, real_c, places=0)

        real_e = 0
        ellipse = self.blob_ellipse.ellipse_fit_residual_mean
        self.assertAlmostEqual(ellipse, real_e, places=0)

    def test_perimeter_convex_hull(self):
        real_c = math.pi * 100
        circle = self.blob_circle.perimeter_convex_hull
        self.assertAlmostEqual(circle, real_c, places=-1)

    def test_pixel_intensities(self):
        coords = np.where(self.im_dist_c > 0)
        real_c = self.im_dist_c[coords]
        circle = self.blob_circle_dist.pixel_intensities
        self.assertEqual(np.all(real_c == circle), True)

    def test_pixel_intensity_mean(self):
        coords = np.where(self.im_dist_c > 0)
        real_c = np.mean(self.im_dist_c[coords])
        circle = self.blob_circle_dist.pixel_intensity_mean
        self.assertEqual(np.all(real_c == circle), True)

    def test_pixel_intensity_median(self):
        coords = np.where(self.im_dist_c > 0)
        real_c = np.median(self.im_dist_c[coords])
        circle = self.blob_circle_dist.pixel_intensity_median
        self.assertEqual(np.all(real_c == circle), True)

    def test_pixel_intensity_std(self):
        coords = np.where(self.im_dist_c > 0)
        real_c = np.std(self.im_dist_c[coords])
        circle = self.blob_circle_dist.pixel_intensity_std
        self.assertEqual(np.all(real_c == circle), True)

    def test_pixel_kurtosis(self):
        coords = np.where(self.im_dist_c > 0)
        real_c = kurtosis(self.im_dist_c[coords], fisher=True, bias=False)
        circle = self.blob_circle_dist.pixel_kurtosis
        self.assertEqual(circle, real_c)

    def test_pixelskew(self):
        coords = np.where(self.im_dist_c > 0)
        real_c = skew(self.im_dist_c[coords], bias=False, nan_policy='omit')
        circle = self.blob_circle_dist.pixel_skew
        self.assertEqual(circle, real_c)

    def test_roughness_perimeter(self):
        real_c = 1
        circle = self.blob_circle.roughness_perimeter
        self.assertAlmostEqual(circle, real_c, places=1)

    def test_roughness_surface(self):
        real_c = 0
        circle = self.blob_circle.roughness_surface
        self.assertAlmostEqual(circle, real_c, places=1)

    def test_roundness(self):
        c_num = 4 * math.pi * 50 * 50
        c_den = math.pi * pow(100, 2)
        real_c = c_num/c_den
        circle = self.blob_circle.roundness
        self.assertAlmostEqual(circle, real_c, places=1)

        e_num = 4 * math.pi * 50 * 100
        e_den = math.pi * pow(200, 2)
        real_e = e_num/e_den
        ellipse = self.blob_ellipse.roundness
        self.assertAlmostEqual(ellipse, real_e, places=1)


if __name__ == '__main__':
    unittest.main()
