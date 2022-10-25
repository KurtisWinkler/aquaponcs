import unittest
import contrast_function as cf
import os
import sys

### UNIT TESTS ###

class classTestContrast(unittest.TestCase):
    # setUp and tearDown
#     @classmethod
#     def setUpClass(cls):
        # not sure class variables are necessary here
    
#     @classmethod
#     def tearDownClass(cls):
        
    
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