import unittest
import numpy as np
import cv2 as cv
import blob_class as bc

class TestBlobDetection(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        im = np.zeros((512,512,3), dtype='uint8')
        
        # specify color and thickness for each created blob
        color = (255, 255, 255)
        thickness = -1

        # create ellipse
        center_coordinates = (250, 400)
        axesLength = (100, 50)
        angle = 30
        startAngle = 0
        endAngle = 360
        im_e = im.copy()
        cv.ellipse(im_e, center_coordinates, axesLength, angle,
                                  startAngle, endAngle, color, thickness)
        
        # create circle
        center_coordinates = (150, 200)
        radius = 50
        im_c = im.copy()
        cv.circle(im_c, center_coordinates, radius, color, thickness)
        
        # make blobs
        im_eg = cv.cvtColor(im_e, cv.COLOR_BGR2GRAY)
        im_cg = cv.cvtColor(im_c, cv.COLOR_BGR2GRAY)
        
        contour_e, _ = cv.findContours(im_eg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        contour_c, _ = cv.findContours(im_cg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        
        # create blobs for testing
        cls.blob_ellipse = bc.Blob(contour_e[0], im_e)
        cls.blob_circle = bc.Blob(contour_c[0], im_c)
        
    @classmethod
    def tearDownClass(cls):
        cls.blob_ellipse = None
        cls.blob_cirlce = None

    #***Test aspect_ratio***
    def test_aspect_ratio(self):
        # check within one decimal point
        self.assertAlmostEqual(self.blob_circle.aspect_ratio, 1, places=1)
        self.assertAlmostEqual(self.blob_ellipse.aspect_ratio, 2, places=1)

        
        
        
if __name__ == '__main__':
    unittest.main()