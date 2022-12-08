import unittest
import os
import sys
import numpy as np
import cv2 as cv
from PIL import Image
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(src_path)
import nucleus_contour as nc  # nopep8
import blob_class as bc  # nopep8


class classTestNucContour(unittest.TestCase):
    # setUp and tearDown
    @classmethod
    def setUpClass(cls):

        im_array_1 = np.zeros((512, 512, 3), dtype='uint8')
        im_array_2 = im_array_1.copy()
        cls.im_zeros = im_array_1.copy()

        # create test circle

        center_coordinates = (150, 200)
        radius = 50
        color = (255, 255, 255)
        thickness = -1
        cls.radius = radius
        cv.circle(im_array_1, center_coordinates, radius, color, thickness)

        # random tests

        random_radii = np.random.randint(25, 60, (1, 10))
        rand_idx = np.random.randint(0, 9)
        cls.random_radius = random_radii[rand_idx]
        cv.circle(im_array_2, center_coordinates,
                  cls.random_radius, color,
                  thickness)

        # Find the set contour
        contour = nc.nucleus_contour(im_array_1)

        cls.contour = contour

        cls.circle_blob = bc.Blob(contour, im_array_1)

        # Find the random contour

        rand_contour = nc.nucleus_contour(im_array_2)

        cls.rand_contour = rand_contour

        cls.rand_circle_blob = bc.Blob(rand_contour, im_array_2)

    @classmethod
    def tearDownClass(cls):
        cls.im_zeros = None
        cls.radius = None

    def test_nucleus_contour_perimeter(self):

        # use normal formulas to calculate perimeter of circle

        calc_perim = 2 * np.pi * self.radius
        contour_perim = self.circle_blob.perimeter_crofton

        # 10% threshold for accuracy

        threshold = .1

        self.assertTrue(calc_perim * (1 - threshold)
                        <= contour_perim <=
                        calc_perim * (1 + threshold))

        # random tests
        calc_perim_rand = 2 * np.pi * self.random_radius
        contour_perim_rand = cls.rand_circle_blob.perimeter_crofton

        self.assertTrue(calc_perim_rand * (1 - threshold)
                        <= contour_perim_rand <=
                        calc_perim_rand * (1 + threshold))

    def test_nucleus_contour_area(self):

        # normal formula for area of circle

        calc_area = np.pi * self.radius**2
        contour_area = self.circle_blob.area_filled

        # 10% threshold for accuracy

        threshold = .1

        self.assertTrue((1 - threshold) * calc_area
                        <= contour_area <=
                        calc_area * (1 + threshold))

        # random tests
        calc_area_rand = np.pi * self.random_radius**2
        contour_area_rand = cls.rand_circle_blob.area_filled

        self.assertTrue(calc_area_rand * (1 - threshold)
                        <= contour_area_rand <=
                        calc_area_rand * (1 + threshold))
