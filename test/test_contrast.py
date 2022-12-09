import unittest
import os
import sys
import numpy as np
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(src_path)
import contrast_functions as cf  # nopep8


class classTestContrast(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.im1 = np.array([1, 2, 3, 4], dtype=np.uint8)
        cls.im2 = np.array([[1, 2], [3, 4]], dtype=np.uint8)

    @classmethod
    def tearDownClass(cls):
        cls.im1 = None
        cls.im2 = None

    def test_min_max_rescale_error(self):
        # TypeError if image (input 1) not list or numpy array
        self.assertRaises(TypeError, cf.min_max_rescale, 'image')

    def test_min_max_rescale_fixed(self):
        # should be rescaled to be between 0 and 255
        test1 = cf.min_max_rescale(self.im1)
        real1 = np.array([0, 85, 170, 255])

        test2 = cf.min_max_rescale(self.im2)
        real2 = np.array([[0, 85], [170, 255]])

        self.assertTrue(np.all(real1 == test1))
        self.assertTrue(np.all(real2 == test2))

    def test_percentile_rescale_error(self):
        # TypeError if image (input 1) not list or numpy array
        self.assertRaises(TypeError, cf.min_max_rescale, 'image')

        # TypeError if min_percentile (input 2) not int or float
        self.assertRaises(TypeError, cf.min_max_rescale, self.im1, 'a', 10)

        # TypeError if max_percentile (input 3) not int or float
        self.assertRaises(TypeError, cf.min_max_rescale, self.im1, 2.5, 'a')

    def test_percentile_rescale_fixed(self):

        test1 = cf.percentile_rescale(self.im1, 0, 100)
        real1 = np.array([0, 85, 170, 255])

        # scale from 0-2.5
        test2 = cf.percentile_rescale(self.im1, 0, 50)
        real2 = np.array([0, 170, 255, 255])

        # scale from 2.5-4
        test3 = cf.percentile_rescale(self.im2, 50, 100)
        real3 = np.array([[0, 0], [85, 255]])

        self.assertTrue(np.all(real1 == test1))
        self.assertTrue(np.all(real2 == test2))
        self.assertTrue(np.all(real3 == test3))


if __name__ == '__main__':
    unittest.main()
