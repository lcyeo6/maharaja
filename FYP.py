# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 01:15:44 2019

@author: User
"""

#import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import re, nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

file = "tripadvisor_co_uk-travel_restaurant_reviews_sample.xlsx"

dataset = pd.read_excel(file, sheet_name = "Sheet_1")

# Shows the column headings
#print("Column headings:")
#print(dataset.columns)

# Shows the whole row of "uniq_id", including index number
#print(dataset["uniq_id"])

# Shows the whole row of "uniq_id", excluding index number
#for i in dataset.index:
#    print(dataset["uniq_id"][i])

#uniq_id = dataset["uniq_id"]
#url = dataset["url"]

stop_words = set(stopwords.words('english'))
wordnet_lemmatizer = WordNetLemmatizer()

# Remove rows if NaN in specific column, in this case "review_text"
dataset = dataset.dropna(subset = ["review_text", "rating"])

def normalizer(dataset):
    only_letters = re.sub("[^a-zA-Z]", " ", dataset) 
    tokens = nltk.word_tokenize(only_letters)
    lower_case = [l.lower() for l in tokens]
    filtered_result = list(filter(lambda l: l not in stop_words, lower_case))
    lemmas = [wordnet_lemmatizer.lemmatize(t) for t in filtered_result]
    return lemmas

dataset["normalized_review_text"] = dataset.review_text.apply(normalizer)
