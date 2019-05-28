"""

@description:
    
    Test Cases for MPQA & SentiWordNet

@author: 
    
    Wan Chee Tin
    Yeo Lin Chung
    
"""

import unittest
import rule_based

class Test_RuleBased(unittest.TestCase):
    
    def test_actual_sentiment_mpqa(self):
        self.assertEqual(rule_based.actual_sentiment_mpqa("1.0 of 5 bubbles"), 1)
        self.assertEqual(rule_based.actual_sentiment_mpqa("2 of 5 bubbles"), 2)
        self.assertEqual(rule_based.actual_sentiment_mpqa("3.0 of 5 bubbles"), 3)
        self.assertEqual(rule_based.actual_sentiment_mpqa("4 of 5 stars"), 4)
        self.assertEqual(rule_based.actual_sentiment_mpqa("5 of 5 bubbles"), 5)
        
    def test_predict_sentiment_mpqa(self):
        self.assertEqual(rule_based.predict_sentiment_mpqa([("Food", "n"), ("awful", "a"), ("bad", "a")]), 1)
        self.assertEqual(rule_based.predict_sentiment_mpqa([("Service", "n"), ("deteriorate", "a"), ("dirty", "a")]), 2)
        self.assertEqual(rule_based.predict_sentiment_mpqa([("Food", "n"), ("bad", "a"), ("great", "a")]), 3)
        self.assertEqual(rule_based.predict_sentiment_mpqa([("Service", "n"), ("elevate", "a"), ("fresh", "a")]), 4)
        self.assertEqual(rule_based.predict_sentiment_mpqa([("Food", "n"), ("delicious", "a"), ("fantastic", "a")]), 5)
        
    def test_actual_sentiment_combined(self):
        self.assertEqual(rule_based.actual_sentiment_combined("1.0 of 5 bubbles"), "negative")
        self.assertEqual(rule_based.actual_sentiment_combined("2 of 5 bubbles"), "negative")
        self.assertEqual(rule_based.actual_sentiment_combined("3.0 of 5 bubbles"), "neutral")
        self.assertEqual(rule_based.actual_sentiment_combined("4 of 5 stars"), "positive")
        self.assertEqual(rule_based.actual_sentiment_combined("5 of 5 bubbles"), "positive")
        
    def test_predict_sentiment_combined(self):
        self.assertEqual(rule_based.predict_sentiment_combined([("Food", "n"), ("awful", "a"), ("bad", "a")]), "negative")
        self.assertEqual(rule_based.predict_sentiment_combined([("Service", "n"), ("deteriorate", "a"), ("dirty", "a")]), "negative")
        self.assertEqual(rule_based.predict_sentiment_combined([("Food", "n"), ("bad", "a"), ("great", "a")]), "neutral")
        self.assertEqual(rule_based.predict_sentiment_combined([("Service", "n"), ("elevate", "a"), ("fresh", "a")]), "positive")
        self.assertEqual(rule_based.predict_sentiment_combined([("Food", "n"), ("delicious", "a"), ("fantastic", "a")]), "positive")
        
    def test_predict_sentiment_swn(self):
        self.assertEqual(rule_based.predict_sentiment_swn([("Food", "n"), ("awful", "a"), ("bad", "a")]), "negative")
        self.assertEqual(rule_based.predict_sentiment_swn([("Service", "n"), ("deteriorate", "a"), ("dirty", "a")]), "negative")
        self.assertEqual(rule_based.predict_sentiment_swn([("Food", "n"), ("awful", "a"), ("good", "a")]), "negative")
        self.assertEqual(rule_based.predict_sentiment_swn([("Service", "n"), ("elevate", "a"), ("fresh", "a")]), "negative")
        self.assertEqual(rule_based.predict_sentiment_swn([("Food", "n"), ("delicious", "a"), ("fantastic", "a")]), "positive")
        
if __name__ == '__main__':
    unittest.main()