"""
@description:
    
    Pre-process an excel file

@author: 
    
    Wan Chee Tin
    Yeo Lin Chung
    
@update:

    1. (06/03/2019) 2 rows of data were removed from the Excel file because both consist of date in 'rating' column.
    2. (12/03/2019) Added documentations for each function.
    
"""

#import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
#import collections
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

def read_excel(filename):
    dataset = pd.read_excel(filename, sheet_name = "Sheet_1")
    return dataset

def pre_process(dataset):
    # Remove unwanted columns
    dataset = dataset.drop(["uniq_id", "url", "restaurant_location", "category", "review_date", "author", "author_url", "location", "visited_on"], axis = 1)
    
    # Remove rows if NaN exists in specific column, in this case "review_text" & "rating" columns
    dataset = dataset.dropna(subset = ["review_text", "rating"])
    
    # Generate a list of stop words such as {'i', 'a', 'the', ...}
    stop_words = set(stopwords.words("english"))
    # To add additional words into the list, just use the append method as follows.
    # stopwords.append(<new_word>)
    
    lemmatizer = WordNetLemmatizer()
    
    def text_normalization(data):
        # Replace unknown characters and numbers with a space
        # [^a-zA-Z] will match any character except lower-case and upper-case letters
        letters = re.sub("[^a-zA-Z]", " ", data)
        
        ## Use nltk.sent_tokenize for splitting the data into sentences
        # Split data into words
        tokens = nltk.word_tokenize(letters)
        
        # Convert tokenized words into tokenized lower-case words
        small_letters = []
        for i in tokens:
            small_letters.append(i.lower())
        
        # Remove stop words 
        filtered = list(filter(lambda l: l not in stop_words, small_letters))
        
        # Lemmatize using WordNet's built-in function
        lemmatized_words = []
        for j in filtered:
            lemmatized_words.append(lemmatizer.lemmatize(j, pos = "v"))
            
        return lemmatized_words
    
    # Apply text_normalization function
    dataset["normalized_review_text"] = dataset.review_text.apply(text_normalization)

    return dataset

def ngrams(data):
    # Generate sequences of normalized words beginning from distinct elements of the list of normalized words
    # The zip function takes the sequences as a list of inputs
    # bigram = A sequence of two adjacent words
    bigram = []
    for i in list(zip(data, data[1:])):
        bigram.append(' '.join(i))
    
    # Generate sequences of normalized words beginning from distinct elements of the list of normalized words
    # The zip function takes the sequences as a list of inputs
    # trigram = A sequence of three adjacent words
    trigram = []
    for j in list(zip(data, data[1:], data[2:])):
        trigram.append(' '.join(j))
    
    return bigram + trigram

# Read data from Excel file
dataset = read_excel("tripadvisor_co_uk-travel_restaurant_reviews_sample.xlsx")

# Data cleaning & pre-processing
filtered_dataset = pre_process(dataset)
filtered_dataset["ngrams"] = filtered_dataset.normalized_review_text.apply(ngrams)


# Create a new data frame to combine same reviews in it
#df2 = pd.DataFrame(columns=["restaurant_id", "name", "number", "total_n_grams"])
#df2 = df2.append({'restaurant_id': df.restaurant_id[0], 'name': df.name[0], 'number': 1, 'total_n_grams': df.n_grams[0]}, ignore_index=True)

# Combine same restaurant's reviews into one for n_grams columns
#for i in range(1, 16):
#    inside = False
#    smthg = None
#    for j in range(len(df2)):
#        if df.restaurant_id[i] == df2.restaurant_id[j]:
#            inside = True
#            smthg = j
#    if inside == True:
#        # The restaurant is in the 2nd data frame.
#        df2.number[smthg] += 1
#        df2.total_n_grams[smthg] += df.n_grams[i]
#    if inside == False:
#        # The restaurant is NOT the 2nd data frame.
#        df2 = df2.append({'restaurant_id': df.restaurant_id[i], 'name': df.name[i], 'number': 1, 'total_n_grams': df.n_grams[i]}, ignore_index=True)        






# Counting the frequency of the words
#def count_words(dataset):
#    counts = collections.Counter()
#    for i in dataset:
#        counts[i] +=1
#    return counts
#df["counts"] = df["n_grams"].apply(count_words)

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










