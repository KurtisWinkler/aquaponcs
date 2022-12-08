import unittest
import numpy as np
import cv2 as cv
from scipy import ndimage as ndi
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(src_path)
import blob_class as bc # nopep8
import blob_detection as bd # nopep8


class TestBlobDetection(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        im = np.zeros((512, 512, 3), dtype='uint8')
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
        cv.circle(im, center_coordinates, radius, color, thickness)

        # create circle 2
        center_coordinates = (350, 150)
        radius = 35
        cv.circle(im, center_coordinates, radius, color, thickness)

        cls.im_shapes = im.copy()

        # find contours and make Blobs of each shape
        im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        contours, _ = cv.findContours(im_gray,
                                      cv.RETR_EXTERNAL,
                                      cv.CHAIN_APPROX_NONE)
        cv.drawContours(im, contours, -1, (0, 255, 0), 2, cv.LINE_8)

        cls.contours = contours
        cls.im_contours = im.copy()

        # create blob list for testing
        cls.blobs = [bc.Blob(contour, im) for contour in contours]

        # Create new image for overlapping blobs
        im2 = np.zeros((512, 512, 3), dtype='uint8')

        # create circle 1
        center_coordinates = (225, 200)
        radius = 60
        cv.circle(im2, center_coordinates, radius, color, thickness)

        # create overlapping circle 2
        center_coordinates = (300, 150)
        radius = 75
        cv.circle(im2, center_coordinates, radius, color, thickness)

        cls.im_overlap = cv.cvtColor(im2, cv.COLOR_BGR2GRAY).copy()

        cls.im_maxima = ndi.distance_transform_edt(cls.im_overlap)

        contour, _ = cv.findContours(cls.im_overlap,
                                     cv.RETR_EXTERNAL,
                                     cv.CHAIN_APPROX_NONE)
        cls.overlap_blob = bc.Blob(contour[0], cls.im_maxima)

        # Create new image for blobs that are inside each other
        im3 = np.zeros((512, 512), dtype='uint8')

        # Inside circle 1
        center_coordinates = (250, 200)
        radius = 35
        cv.circle(im3, center_coordinates, radius, color, thickness)

        # Inside circle 2
        center_coordinates = (200, 300)
        radius = 50
        cv.circle(im3, center_coordinates, radius, color, thickness)

        # Inside circle contours
        contours_in, _ = cv.findContours(im3,
                                         cv.RETR_EXTERNAL,
                                         cv.CHAIN_APPROX_NONE)

        # Outside circle
        center_coordinates = (225, 250)
        radius = 130
        cv.circle(im3, center_coordinates, radius, color, thickness)

        # Outside circle contours
        contours_out, _ = cv.findContours(im3,
                                          cv.RETR_EXTERNAL,
                                          cv.CHAIN_APPROX_NONE)

        # All contours
        contours_in_out = contours_in + contours_out

        # Blobs (idx 2 is outside blob)
        cls.blobs_in_out = [bc.Blob(con, im3) for con in contours_in_out]

    @classmethod
    def tearDownClass(cls):
        cls.im_zeros = None
        cls.im_shapes = None
        cls.im_contours = None
        cls.contours = None
        cls.blobs = None
        cls.im_overlap = None
        cls.im_maxima = None
        cls.overlap_blob = None
        cls.blobs_in_out = None

    # ***Test flatten***
    def test_flatten_error(self):
        # TypeError if L (input 1) not list
        self.assertRaises(TypeError, bd.flatten, 'blob')

    def test_flatten_fixed(self):
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

    # ***Test blob_im***
    def test_blob_im_error(self):
        # TypeError if im (input 1) not list or numpy array
        self.assertRaises(TypeError, bd.blob_im, 'blob', self.blobs)

        # TypeError if blobs (input 2) not all blobs
        self.assertRaises(TypeError, bd.blob_im, self.im_zeros,
                          [self.blobs, 'a'])

    def test_blob_im_fixed(self):
        real1 = self.im_contours
        test1 = bd.blob_im(self.im_shapes, self.blobs)
        self.assertEqual(np.all(real1 == test1), True)
        self.assertNotEqual(np.all(self.im_shapes == test1), True)

    # ***Test get_maxima***
    def test_get_maxima_error(self):
        # TypeError if image (input 1) not numpy array or blob
        self.assertRaises(TypeError, bd.get_maxima, 'blob', 20, 0.8)

        # TypeError if distance (input 2) not int
        self.assertRaises(TypeError, bd.get_maxima, self.overlap_blob,
                          10.5, 0.8)

        # TypeError if threshold (input 3) not int or float
        self.assertRaises(TypeError, bd.get_maxima, self.overlap_blob,
                          10.5, 'a')

        # IndexError if distance < 0
        self.assertRaises(IndexError, bd.get_maxima, self.overlap_blob,
                          -1, 0.8)

        # IndexError if threshold not 0 <= threshold <= 1
        self.assertRaises(IndexError, bd.get_maxima, self.overlap_blob,
                          10, 1.2)

    def test_get_maxima_fixed(self):
        # local maxima points (centers of overlapping blobs)
        real = [[300, 150], [225, 200]]

        # test blob input
        test1 = bd.get_maxima(self.overlap_blob, 20, 0.8)
        self.assertEqual(real, test1)

        # test image input
        test2 = bd.get_maxima(self.im_maxima, 20, 0.8)
        self.assertEqual(real, test2)

    # ***Test get_contours***
    def test_get_contours_error(self):
        # TypeError if image (input 1) not list or numpy array
        self.assertRaises(TypeError, bd.get_contours, 'blob', 20, 255, 10)

        # TypeError if thresh_min (input 2) not int
        self.assertRaises(TypeError, bd.get_contours, self.im_shapes,
                          8.5, 255, 10)

        # TypeError if thresh_max (input 3) not int
        self.assertRaises(TypeError, bd.get_contours, self.im_shapes,
                          20, 245.5, 10)

        # TypeError if thresh_step (input 4) not int
        self.assertRaises(TypeError, bd.get_contours, self.im_shapes,
                          20, 255, 8.5)

        # IndexError if not 0 <= thresh_min <= thresh_max <= 255
        self.assertRaises(IndexError, bd.get_contours, self.im_shapes,
                          -5, 255, 10)
        self.assertRaises(IndexError, bd.get_contours, self.im_shapes,
                          65, 55, 10)
        self.assertRaises(IndexError, bd.get_contours, self.im_shapes,
                          20, 265, 10)

        # IndexError if not 1 <= thresh_step <= thresh_max
        self.assertRaises(IndexError, bd.get_contours, self.im_shapes,
                          20, 255, -5)
        self.assertRaises(IndexError, bd.get_contours, self.im_shapes,
                          20, 110, 125)

    def test_get_contours_fixed(self):
        # number of contours that should be found
        real = 4 * 5  # num blobs * iterating 5 times to find contours

        im_shapes_gray = cv.cvtColor(self.im_shapes, cv.COLOR_BGR2GRAY)
        test = bd.get_contours(im_shapes_gray, 20, 110, 20)  # iterates 5 times
        self.assertEqual(real, len(test))

    # ***Test segment_contours***
    def test_segment_contours_error(self):
        # TypeError if binary_image (input 1) not list of numpy array
        self.assertRaises(TypeError, bd.segment_contours, 'blob', 10)

        # TypeError if min_distance (input 2) not int
        self.assertRaises(TypeError, bd.segment_contours,
                          self.im_overlap, 10.5)

        # IndexError if min_distance < 0
        self.assertRaises(IndexError, bd.segment_contours, self.im_overlap, -4)

    def test_segment_contours_fixed(self):
        # number of contours that should be found
        real = 2  # overlap_blob should segment into 2 blobs

        test = bd.segment_contours(self.im_overlap, 50)
        self.assertEqual(real, len(test))

    # ***Test organize_contours***
    def test_organize_contours_error(self):
        im = self.im_shapes
        con = self.contours
        pts = [[1, 2], [5, 6]]
        # TypeError if contours (input 1) not list, numpy array, or tuple
        self.assertRaises(TypeError, bd.organize_contours, 'con', pts, im, 10)

        # TypeError if coords (input 2) not list, numpy array, or tuple
        self.assertRaises(TypeError, bd.organize_contours, con, 'pts', im, 10)

        # TypeError if image (input 3) not numpy ndarray
        self.assertRaises(TypeError, bd.organize_contours, con, pts, 'im', 10)

        # TypeError if min_distance (input 4) not int
        self.assertRaises(TypeError, bd.organize_contours, con, pts, im, 10.5)

        # IndexError if min_distance >= 0
        self.assertRaises(IndexError, bd.organize_contours, con, pts, im, -1)

    def test_organize_contours_fixed(self):
        im = self.im_shapes
        con = self.contours
        pts = [[250, 400], [450, 325], [150, 200], [350, 150], [500, 500]]
        test = bd.organize_contours(con, pts, im, 10)

        # each pt in pts should have one contour except last
        for idx, org in enumerate(test):
            if idx <= 3:
                self.assertEqual(len(org), 1)
            else:
                self.assertEqual(len(org), 0)

    # ***Test maxima_filter***
    def test_maxima_filter_error(self):
        # TypeError if contour (input 1) not list or numpy array
        self.assertRaises(TypeError, bd.maxima_filter, 'blob', [[1, 2]])

        # TypeError if local_maxima (input 2) not list or numpy array
        self.assertRaises(TypeError, bd.maxima_filter, [[1, 2]], 'blob')

        # IndexError if contour (input 1) wrong shape
        self.assertRaises(IndexError, bd.maxima_filter, [1, 2, 3], [[1, 2]])

        # IndexError if local_maxima (input 2) wrong shape
        self.assertRaises(IndexError, bd.maxima_filter, [[1, 2]], [1, 2, 3])

    def test_maxima_filter_fixed(self):
        contours = [blob.contour for blob in self.blobs]
        maxima = [blob.centroid_xy for blob in self.blobs]

        for i in range(len(maxima)):
            real = (np.array([maxima[i]]), np.array([i]))
            test = bd.maxima_filter(contours[i], maxima)
            self.assertEqual(real[0][0][0], test[0][0][0])  # x-coord
            self.assertEqual(real[0][0][1], test[0][0][1])  # y-coord
            self.assertEqual(real[1], test[1])  # index

    # ***Test blob_filter***
    def test_blob_filter_error(self):
        filt1 = [['circularity', 0.8, None]]

        filt2 = [['circularity', 0.8],
                 ['area', 0.5]]

        filt3 = [['circularity', 0.8, None],
                 ['area', 0.5]]

        # TypeError if blob (input 1) not instance of class Blob
        self.assertRaises(TypeError, bd.blob_filter, 'blob', filt1)

        # TypeError if filters (input 2) in not type list
        self.assertRaises(TypeError, bd.blob_filter, self.blobs[0], 'area')

        # IndexError if filters is not right size
        self.assertRaises(IndexError, bd.blob_filter, self.blobs[0], filt2)
        self.assertRaises(IndexError, bd.blob_filter, self.blobs[0], filt3)

    def test_blob_filter_fixed(self):
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

    # ***Test similar_filter***
    def test_similar_filter_error(self):
        param1 = [['circularity', 0.2]]

        param2 = [['circularity', 0.2, 0.5],
                  ['area', 0.5, 0.8]]

        param3 = [['circularity', 0.8, None],
                  ['area', 0.5]]

        # TypeError if blob_list (input 1) not list
        self.assertRaises(TypeError, bd.similar_filter, 'blob', param1)

        # TypeError if blob_list (input 1) not all blobs
        self.assertRaises(TypeError, bd.similar_filter,
                          [self.blobs, 'a'], param1)

        # TypeError if filters (input 2) in not type list
        self.assertRaises(TypeError, bd.similar_filter, self.blobs, 'area')

        # TypeError if num (input 3) in not type int
        self.assertRaises(TypeError, bd.similar_filter,
                          self.blobs, param1, 'a')

        # IndexError if filters is not right size
        self.assertRaises(IndexError, bd.similar_filter, self.blobs, param2)
        self.assertRaises(IndexError, bd.similar_filter, self.blobs, param3)

        # IndexError if num (input 3) in < 1
        self.assertRaises(IndexError, bd.similar_filter, self.blobs, param1, 0)

    def test_similar_filter_fixed(self):
        # param1 should return blobs 1,2,3
        param1 = [['aspect_ratio', 0.2]]
        real1 = self.blobs[1:]
        test1 = bd.similar_filter(self.blobs, param1, num=3)

        # param2 should return blobs 0, 1
        param2 = [['area', 0.2],
                  ['axis_major_length', 0.25]]
        real2 = self.blobs[0:2]
        test2 = bd.similar_filter(self.blobs, param2)

        self.assertEqual(real1, test1)
        self.assertEqual(real2, test2)

    # ***Test outlier_filter***
    def test_outlier_filter_error(self):
        param1 = [['circularity', 0.2, 1]]

        param2 = [['circularity', 0.2],
                  ['area', 0.3]]

        param3 = [['circularity', 0.2, 2],
                  ['area', 0.5]]

        # TypeError if blob_list (input 1) not instance of class Blob
        self.assertRaises(TypeError, bd.outlier_filter, 'blob', param1)

        # TypeError if blob_list (input 1) not all blobs
        self.assertRaises(TypeError, bd.outlier_filter,
                          [self.blobs, 'a'], param1)

        # TypeError if filters (input 2) in not type list
        self.assertRaises(TypeError, bd.outlier_filter, self.blobs, 'area')

        # IndexError if filters is not right size
        self.assertRaises(IndexError, bd.outlier_filter, self.blobs, param2)
        self.assertRaises(IndexError, bd.outlier_filter, self.blobs, param3)

    def test_outlier_filter_fixed(self):
        # param1 should return blobs 1,2,3
        param1 = [['aspect_ratio', 0.2, 1.5]]
        real1 = self.blobs[1:]
        test1 = bd.outlier_filter(self.blobs, param1)

        # param2 should return blobs 2, 3
        param2 = [['aspect_ratio', 0.2, 1.5],
                  ['circularity', 0.2, 1]]
        real2 = self.blobs[2:]
        test2 = bd.outlier_filter(self.blobs, param2)

        self.assertEqual(real1, test1)
        self.assertEqual(real2, test2)

    # ***Test blob_best***
    def test_blob_best_error(self):
        param1 = [['circularity', 'max', 1]]

        param2 = [['circularity', 0.2],
                  ['area', 0.3]]

        param3 = [['circularity', 'max', 2],
                  ['area', 0.5]]

        # TypeError if blob_list (input 1) not instance of list
        self.assertRaises(TypeError, bd.blob_best, 'blob', param1)

        # TypeError if blob_list (input 1) not all blobs
        self.assertRaises(TypeError, bd.blob_best, [self.blobs, 'a'], param1)

        # TypeError if filters (input 2) in not type list
        self.assertRaises(TypeError, bd.blob_best, self.blobs, 'area')

        # IndexError if filters is not right size
        self.assertRaises(IndexError, bd.blob_best, self.blobs, param2)
        self.assertRaises(IndexError, bd.blob_best, self.blobs, param3)

    def test_blob_best_fixed(self):
        # param1 should return blob 0
        param1 = [['aspect_ratio', 'max', 1]]
        real1 = self.blobs[0]
        test1 = bd.blob_best(self.blobs, param1)

        # param2 should return blob 3
        param2 = [['area', 'min', 2],
                  ['circularity', 1, 1]]
        real2 = self.blobs[3]
        test2 = bd.blob_best(self.blobs, param2)

        self.assertEqual(real1, test1)
        self.assertEqual(real2, test2)

        # ***Test final_blobs_filter***
    def test_final_blobs_filter_error(self):
        # TypeError if blobs (input 1) not instance of list
        self.assertRaises(TypeError, bd.final_blobs_filter, 'blobs')

        # TypeError if blobs (input 1) not all blobs
        self.assertRaises(TypeError, bd.final_blobs_filter, [self.blobs, 'a'])

    def test_final_blobs_filter_fixed(self):
        # blobs[0] and blobs[1] are inside blobs[2]
        blobs = self.blobs_in_out
        real = blobs[0:2]
        test = bd.final_blobs_filter(blobs)

        self.assertEqual(real, test)


if __name__ == '__main__':
    unittest.main()
