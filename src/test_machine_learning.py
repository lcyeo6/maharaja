"""

@description:
    
    Test Cases for SVM & Random Forest

@author: 
    
    Wan Chee Tin
    Yeo Lin Chung
    
"""

import unittest
import machine_learning

class Test_MachineLearning(unittest.TestCase):
    
    def test_class_equity(self):
        self.assertEqual(machine_learning.class_equity(),)
    
if __name__ == '__main__':
    unittest.main()