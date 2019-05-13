"""

@description:
    
    Lexicon & SVM-based Sentiment Analysis

@author: 
    
    Wan Chee Tin
    Yeo Lin Chung
    
"""

import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def read_excel(filename):
    
    dataset = pd.read_excel(filename, sheet_name = "Sheet1")
    
    return dataset

# Read the data from Excel file
dataset = read_excel("MPQA_Dataset.xlsx")

temp = pd.DataFrame(columns = ["name", "title"])
