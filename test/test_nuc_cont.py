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

        im_array = np.zeros((512, 512, 3), dtype='uint8')
        cls.im_zeros = im_array.copy()

        # create test circle

        center_coordinates = (150, 200)
        radius = 50
        color = (255, 255, 255)
        thickness = -1
        cls.radius = radius
        cv.circle(im_array, center_coordinates, radius, color, thickness)

        # create test image
        src_path = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), '../'))
        sys.path.append(src_path)
        im = Image.fromarray(im_array)
        im.save('test_image.jpeg')

        # Find the contour
        output_name, contour, img = nc.nucleus_contour('test_image.jpeg',
                                                       'test_cont.jpg',
                                                       'test_circle.jpg')

        cls.contour = contour

        cls.circle_blob = bc.Blob(contour, img)

    @classmethod
    def tearDownClass(cls):
        cls.im_zeros = None
        cls.radius = None
        os.remove('test_image.jpeg')
        os.remove('test_cont.jpg')
        os.remove('test_circle.jpg')

    def test_nucleus_contour_perimeter(self):
        # use normal formulas to calculate perimeter of circle

        calc_perim = 2 * np.pi * self.radius
        contour_perim = self.circle_blob.perimeter_crofton

        # 10% threshold for accuracy

        threshold = .1

        self.assertTrue(calc_perim * (1 - threshold)
                        <= contour_perim <=
                        calc_perim * (1 + threshold))

    def test_nucleus_contour_area(self):
        # normal formula for area of circle
        calc_area = np.pi * self.radius**2
        contour_area = self.circle_blob.area_filled

        # 10% threshold for accuracy

        threshold = .1

        self.assertTrue((1 - threshold) * calc_area
                        <= contour_area <=
                        calc_area * (1 + threshold))
