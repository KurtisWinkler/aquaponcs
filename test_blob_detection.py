import unittest
import numpy as np
import cv2 as cv
import blob_class as bc
import blob_detection as bd

class TestBlobDetection(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        im = np.zeros((512,512,3), dtype='uint8')
        cls.im_zeros = im.copy()
        
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
        
        # create messy-shaped blob
        start_point = (400, 400)
        end_point = (500, 250)
        im = cv.rectangle(im, start_point, end_point, color, thickness)
        start_point = (350, 350)
        end_point = (400, 300)
        im = cv.rectangle(im, start_point, end_point, color, thickness)

        # create circle 1
        center_coordinates = (150, 200)
        radius = 50
        color = (255, 255, 255)
        thickness = -1
        cv.circle(im, center_coordinates, radius, color, thickness)
        
        # create circle 2
        center_coordinates = (350, 150)
        radius = 35
        color = (255, 255, 255)
        thickness = -1
        cv.circle(im, center_coordinates, radius, color, thickness)

        cls.im_shapes = im.copy()
        
        # find contours and make Blobs of each shape
        im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        contours, _ = cv.findContours(im_gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        cv.drawContours(im, contours, -1, (0, 255, 0), 2, cv.LINE_8)
        
        cls.contours = contours
        cls.im_contours = im.copy()
        
        # create blob list for testing
        cls.blobs = [bc.Blob(contour, im) for contour in contours]
        # for blob in cls.blobs:
        #     print(blob.area)

    @classmethod
    def tearDownClass(cls):
        cls.im_zeros = None
        cls.im_shapes = None
        cls.im_contours = None
        cls.contours = None
        cls.blobs = None

    #***Test flatten***
    def test_error_flatten(self):
        # TypeError if L (input 1) not list
        self.assertRaises(TypeError, bd.flatten, 'blob')
        
    def test_fixed_flatten(self):
        L1 = [1, 5, 4]
        real1 = [1, 5, 4]
        test1 = bd.flatten(L1)
        
        L2 = [[1, 2, 3], [[4, 5, 6]], [7, [8, 9]]]
        real2 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        test2 = bd.flatten(L2)
        
        L3 = [['abc', 'a', 2], [2.5, 'b']]
        real3 = ['abc', 'a', 2, 2.5, 'b']
        test3 = bd.flatten(L3)
        
        self.assertEqual(real1, test1)
        self.assertEqual(real2, test2)
        self.assertEqual(real3, test3)
        
    #***Test blob_im***
    def test_error_blob_im(self):
        # TypeError if im (input 1) not list or numpy array
        self.assertRaises(TypeError, bd.blob_im, 'blob', self.blobs)
        
        # TypeError if blobs (input 2) not all blobs
        self.assertRaises(TypeError, bd.blob_im, self.im_zeros, [self.blobs, 'a'])
        
    def test_fixed_blob_im(self):
        real1 = self.im_contours
        test1 = bd.blob_im(self.im_shapes, self.blobs)
        self.assertEqual(np.all(real1==test1), True)
        self.assertNotEqual(np.all(self.im_shapes==test1), True)
    
    # ***Test blob_filter***
    def test_error_blob_filter(self):
        filt1 = [['circularity', 0.8, None]]
        
        filt2 = [['circularity', 0.8],
                 ['area', 0.5]]
        
        filt3 = [['circularity', 0.8, None],
                 ['area', 0.5]]
        
        # TypeError if blob not instance of class Blob
        self.assertRaises(TypeError, bd.blob_filter, 'blob', filt1)
        self.assertRaises(TypeError, bd.blob_filter, self.blobs[0], 'area')
        
        # IndexError if filters is not right size
        self.assertRaises(IndexError, bd.blob_filter, self.blobs[0], filt2)
        self.assertRaises(IndexError, bd.blob_filter, self.blobs[0], filt3)
        
    def test_fixed_blob_filter(self):
        # filt1 keeps blobs 0,2,3
        filt1 = [['circularity', 0.8, None]]
        real1 = [True, False, True, True]
        test1 = [bd.blob_filter(blob, filt1) for blob in self.blobs]
        
        # filt2 keeps blobs 0,2
        filt2 = [['circularity', 0.8, None],  # keep blobs 0,2,4
                 ['area', 5000, 16000]]  # keep blobs 0,2
        real2 = [True, False, True, False]
        test2 = [bd.blob_filter(blob, filt2) for blob in self.blobs]
        
        self.assertEqual(real1, test1)
        self.assertEqual(real2, test2)

if __name__ == '__main__':
    unittest.main()
