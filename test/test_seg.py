import unittest
import image_segmentation as isg

### UNIT TESTS ###

class classTestSegmentation(unittest.TestCase):
    # @classmethod
    # def setUpClass(cls):
    #
    # @classmethod
    # def tearDownClass(cls):

    def test_imageSegmentation(self):
        # test to make sure the file exists
        if isg.imageSegmentation('siL_pre_6-slice2-adjusted.tif'):
            self.assertRaise(FileNotFoundError)


if __name__ == '__main__':
    unittest.main()
