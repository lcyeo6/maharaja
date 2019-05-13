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

select_data = pd.Series([])

for index, row in dataset.iterrows():
    if (row[8] == 3) or (row[8] == row[9]):
        select_data[index] = 1
    elif row[8] != row[9]:
        select_data[index] = 0

dataset.insert(10, "one_or_zero", select_data)

dataset = dataset[dataset.one_or_zero != 0]
