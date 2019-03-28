# -*- coding: utf-8 -*-
"""

@author: 
    
    Wan Chee Tin
    Yeo Lin Chung
    
"""

import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
import re, nltk
import collections
#from pandas import ExcelWriter
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import ngrams

def read_excel(file):
    dataset = pd.read_excel(file, sheet_name = "Sheet_1")
    return dataset

def pre_process(dataset):
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
    return dataset

def ngrams(dataset):
    bi_gram = [' '.join(t) for t in list(zip(dataset, dataset[1:]))]
    tri_gram = [' '.join(t) for t in list(zip(dataset, dataset[1:], dataset[2:]))]
    return bi_gram + tri_gram

def count_words(dataset):
    counts = collections.Counter()
    for i in dataset:
        counts[i] +=1
    return counts
    


#Get data
data = read_excel("review_data_test.xlsx")
#Data cleaning & Pre-processing
df = pre_process(data)
#N_grams the normalized data
df["n_grams"] = df.normalized_review_text.apply(ngrams)
# Counting the frequency of the words DO LATER.. NOT NOW.. move down
df["counts"] = df["n_grams"].apply(count_words)
#Create a new data frame to combine same reviews in it
df2 = pd.DataFrame(columns=["restaurant_id", "name", "number", "total_n_grams"])
df2 = df2.append({'restaurant_id': df.restaurant_id[0], 'name': df.name[0], 'number': 1, 'total_n_grams': df.n_grams[0]}, ignore_index=True)
#Combine same restaurant's reviews into one for n_grams columns
for i in range(1, 10):
    inside = False
    smthg = None
    for j in range(len(df2)):
        if df.restaurant_id[i] == df2.restaurant_id[j]:
            inside = True
            smthg = j
    if inside == True:
        # The restaurant is in the 2nd data frame.
        df2.number[smthg] +=1
        df2.total_n_grams[smthg] += df.n_grams[i]
    if inside == False:
        # The restaurant is NOT the 2nd data frame.
        df2 = df2.append({'restaurant_id': df.restaurant_id[i], 'name': df.name[i], 'number': 1, 'total_n_grams': df.n_grams[i]}, ignore_index=True)
            

#from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import accuracy_score
#from sklearn.model_selection import train_test_split
#
#target = [1 if i < 12500 else 0 for i in range(25000)]
#
#X_train, X_val, y_train, y_val = train_test_split(
#    X, target, train_size = 0.75
#)
#
#for c in [0.01, 0.05, 0.25, 0.5, 1]:
#    
#    lr = LogisticRegression(C=c)
#    lr.fit(X_train, y_train)
#    print ("Accuracy for C=%s: %s" 
#           % (c, accuracy_score(y_val, lr.predict(X_val))))










