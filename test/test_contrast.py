import unittest
import os
import sys
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(src_path)
import contrast_functions as cf # nopep8

class classTestContrast(unittest.TestCase):

    def test_min_max_rescale(self):
        # test to make sure the file exists
        if cf.min_max_rescale('bad_file_path.csv'):
            self.assertRaise(FileNotFoundError)

    def test_percentile_rescale(self):
        # test to make sure the file exists
        if cf.percentile_rescale('bad_file_path.csv', .05, .99):
            self.assertRaise(FileNotFoundError)

        # want a positive test for min and max percentiles


if __name__ == '__main__':
    unittest.main()
