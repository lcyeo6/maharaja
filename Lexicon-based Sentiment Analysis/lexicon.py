"""

@description:
    
    Lexicon-based Sentiment Analysis

@author: 
    
    Wan Chee Tin
    Yeo Lin Chung
    
@update:

    
"""

import pandas as pd

def read_excel(filename):
    
    dataset = pd.read_excel(filename, sheet_name = "Sheet1")
    return dataset

# Read the cleaned data from Excel file
dataset = pd.read_excel("preprocessed.xlsx")

# Read the MPQA lexicon file from CSV file
MPQA_Lexicon = pd.read_csv("MPQA/MPQA_Lexicon.csv", header = None)

# Construct a Python dictionary to hold the lexicon
MPQA = dict()

for index, row in MPQA_Lexicon.iterrows():
    MPQA[row[0]] = int(row[1])
    
