import unittest
import numpy as np
import cv2 as cv
import math
from scipy import ndimage as ndi
import os
import sys
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(src_path)
import blob_class as bc  # nopep8
import param_output as po  # nopep8


class TestBlobDetection(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        im = np.zeros((512, 512, 3), dtype='uint8')

        # specify color and thickness for each created blob
        color = (255, 255, 255)
        thickness = -1

        # create ellipse
        center_coordinates = (250, 400)
        axesLength = (100, 50)
        angle = 30
        startAngle = 0
        endAngle = 360
        im = cv.ellipse(im, center_coordinates, axesLength, angle,
                        startAngle, endAngle, color, thickness)

        # create circle 1
        center_coordinates = (150, 200)
        radius = 50
        cv.circle(im, center_coordinates, radius, color, thickness)

        # create circle 2
        center_coordinates = (350, 150)
        radius = 35
        cv.circle(im, center_coordinates, radius, color, thickness)

        cls.im_shapes = im.copy()
        # transfrom im_shapes to create more realistic blobs
        cls.im_dist = ndi.distance_transform_edt(cls.im_shapes)

        # find contours and make Blobs of each shape
        im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        contours, _ = cv.findContours(im_gray,
                                      cv.RETR_EXTERNAL,
                                      cv.CHAIN_APPROX_NONE)

        # create blob list for testing
        cls.blobs = [bc.Blob(contour, cls.im_dist) for contour in contours]

    @classmethod
    def tearDownClass(cls):
        cls.im_shapes = None
        cls.im_dist = None
        cls.blobs = None

    # ***Test flatten***
    def test_build_dict(self):
        test = po.build_dict()
        self.assertEqual(type(test), dict)

    def test_add_value_error(self):
        blob = self.blobs[0]
        # TypeError if blob_params (input 1) not dict
        self.assertRaises(TypeError, po.add_value, 'dict', blob, 3)

        # TypeError if blob (input 2) not Blob
        self.assertRaises(TypeError, po.add_value, {}, 'blob', 3)

        # TypeError if dec (input 3) not int
        self.assertRaises(TypeError, po.add_value, {}, blob, 3.5)

    def test_add_value_fixed(self):
        test = po.build_dict()
        # add ellipse
        po.add_value(test, self.blobs[0], 2)
        # add circle1
        po.add_value(test, self.blobs[1], 2)
        # add circle2
        po.add_value(test, self.blobs[2], 2)

        test1 = float(test['aspect_ratio'][0])
        real1 = 2

        test2 = float(test['circularity'][1])
        real2 = 1

        test3 = float(test['area_filled'][2])
        real3 = math.pi * 35 * 35

        self.assertAlmostEqual(real1, test1, places=1)
        self.assertAlmostEqual(real2, test2, places=1)
        self.assertAlmostEqual(real2, test2, places=1)

    def test_get_params_error(self):
        # TypeError if blobs (input 1) not list or numpy array
        self.assertRaises(TypeError, po.get_params, 'list', 2)

        # TypeError if blobs (input 1) not all blobs
        self.assertRaises(TypeError, po.add_value, [self.blobs, 'blob'], 2)

        # TypeError if dec (input 2) not int
        self.assertRaises(TypeError, po.add_value, self.blobs, 2.5)

    def test_get_params_fixed(self):
        test = po.get_params(self.blobs)

        test1 = float(test['aspect_ratio'][0])
        real1 = 2

        test2 = float(test['circularity'][1])
        real2 = 1

        test3 = float(test['area_filled'][2])
        real3 = math.pi * 35 * 35

        self.assertAlmostEqual(real1, test1, places=1)
        self.assertAlmostEqual(real2, test2, places=1)
        self.assertAlmostEqual(real2, test2, places=1)


if __name__ == '__main__':
    unittest.main()
