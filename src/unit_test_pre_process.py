"""

@description:
    
    Test Cases for Pre-process

@author: 
    
    Wan Chee Tin
    Yeo Lin Chung
    
"""

import unittest
import preprocess
from nltk.corpus import wordnet

class Test_Preprocess(unittest.TestCase):
    
    def test_wordnet_tag(self):
        self.assertEqual(preprocess.wordnet_tag("JJR"), wordnet.ADJ)
        self.assertEqual(preprocess.wordnet_tag("RBR"), wordnet.ADV)
        self.assertEqual(preprocess.wordnet_tag("NNS"), wordnet.NOUN)
        self.assertEqual(preprocess.wordnet_tag("VB"), wordnet.VERB)

    
if __name__ == '__main__':
    unittest.main()