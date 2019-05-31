"""

@description:
    
    Test Cases for MPQA & SentiWordNet

@author: 
    
    Wan Chee Tin
    Yeo Lin Chung
    
"""

import unittest
import lexicon

class Test_RuleBased(unittest.TestCase):
    
    def test_actual_sentiment_combined(self):
        self.assertEqual(lexicon.actual_sentiment_combined("1.0 of 5 bubbles"), "negative")
        self.assertEqual(lexicon.actual_sentiment_combined("2 of 5 bubbles"), "negative")
        self.assertEqual(lexicon.actual_sentiment_combined("3.0 of 5 bubbles"), "neutral")
        self.assertEqual(lexicon.actual_sentiment_combined("4 of 5 stars"), "positive")
        self.assertEqual(lexicon.actual_sentiment_combined("5 of 5 bubbles"), "positive")
        
    def test_predict_sentiment_mpqa(self):
        self.assertEqual(lexicon.predict_sentiment_mpqa([("Food", "n"), ("awful", "a"), ("bad", "a")]), "negative")
        self.assertEqual(lexicon.predict_sentiment_mpqa([("Service", "n"), ("deteriorate", "a"), ("dirty", "a")]), "negative")
        self.assertEqual(lexicon.predict_sentiment_mpqa([("Food", "n"), ("bad", "a"), ("great", "a")]), "neutral")
        self.assertEqual(lexicon.predict_sentiment_mpqa([("Service", "n"), ("elevate", "a"), ("fresh", "a")]), "positive")
        self.assertEqual(lexicon.predict_sentiment_mpqa([("Food", "n"), ("delicious", "a"), ("fantastic", "a")]), "positive")
      
    def test_predict_sentiment_swn(self):
        self.assertEqual(lexicon.predict_sentiment_swn([("Food", "n"), ("awful", "a"), ("bad", "a")]), "negative")
        self.assertEqual(lexicon.predict_sentiment_swn([("Service", "n"), ("deteriorate", "a"), ("dirty", "a")]), "negative")
        self.assertEqual(lexicon.predict_sentiment_swn([("Food", "n"), ("awful", "a"), ("good", "a")]), "negative")
        self.assertEqual(lexicon.predict_sentiment_swn([("Service", "n"), ("elevate", "a"), ("fresh", "a")]), "negative")
        self.assertEqual(lexicon.predict_sentiment_swn([("Food", "n"), ("delicious", "a"), ("fantastic", "a")]), "positive")
        
if __name__ == '__main__':
    unittest.main()