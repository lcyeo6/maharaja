# -*- coding: utf-8 -*-
"""

@author: 
    
    Wan Chee Tin
    Yeo Lin Chung
    
"""

#import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import re, nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

file = "tripadvisor_co_uk-travel_restaurant_reviews_sample.xlsx"

dataset = pd.read_excel(file, sheet_name = "Sheet_1")

# Remove rows if NaN in specific column, in this case "review_text"
dataset = dataset.dropna(subset = ["review_text", "rating"])

stop_words = set(stopwords.words('english'))
wordnet_lemmatizer = WordNetLemmatizer()

def normalizer(dataset):
    letters = re.sub("[^a-zA-Z]", " ", dataset) 
    tokens = nltk.word_tokenize(letters)
    lower_case = [l.lower() for l in tokens]
    filtered_result = list(filter(lambda l: l not in stop_words, lower_case))
    lemmas = [wordnet_lemmatizer.lemmatize(t) for t in filtered_result]
    return lemmas

dataset["normalized_review_text"] = dataset.review_text.apply(normalizer)

# 2 rows of data were removed from the Excel file because both consist of date in 'rating' column

# Get the rating of each review from the dataset
# for i in dataset["rating"]:
#    print(i[0])
