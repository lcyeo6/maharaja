"""

@description:
    
    Test Cases for SVM & Random Forest

@author: 
    
    Wan Chee Tin
    Yeo Lin Chung
    
"""

import unittest
import pre_process

class Test_Preprocess(unittest.TestCase):
    
    def test_class_equity(self):
        self.assertEqual(pre_process.class_equity(),)
    
if __name__ == '__main__':
    unittest.main()