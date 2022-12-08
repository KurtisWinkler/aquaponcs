import unittest
import contrast_functions as cf
import os
import sys


class classTestContrast(unittest.TestCase):

    def test_min_max_rescale(self):
        # test to make sure the file exists
        if cf.min_max_rescale('bad_file_path.csv'):
            self.assertRaise(FileNotFoundError)

    def test_percentile_rescale(self):
        # test to make sure the file exists
        if cf.percentile_rescale('bad_file_path.csv'):
            self.assertRaise(FileNotFoundError)

        # want a positive test for min and max percentiles


if __name__ == '__main__':
    unittest.main()
