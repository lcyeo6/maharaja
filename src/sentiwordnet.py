"""

@description:
    
    Pre-process an excel file

@author: 
    
    Wan Chee Tin
    Yeo Lin Chung
    
"""

import pandas as pd
from preprocess import pre_process
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.corpus import wordnet
from nltk.corpus import sentiwordnet

def actual_sentiment_swn(data):
    
    # Convert star-rating into respective sentiments
    for i in data:
        if int(i[0]) == 4 or int(i[0]) == 5:
            return "positive"
        elif int(i[0]) == 1 or int(i[0]) == 2:
            return "negative"
        else:
            return "neutral"

def predict_sentiment_swn(data):
    
    sentiment_score = 0
    for word, tag in data:
        wordnet_synsets = wordnet.synsets(word, pos = tag)
        if not wordnet_synsets:
            continue
        else:
            wordnet_synset = wordnet_synsets[0]
            sentiwordnet_synset = sentiwordnet.senti_synset(wordnet_synset.name())
            sentiment_score += sentiwordnet_synset.pos_score() - sentiwordnet_synset.neg_score()
    
    if sentiment_score > 0:
        return "positive"
    elif sentiment_score < 0:
        return "negative"
    else:
        return "neutral"
    
# Read the data from Excel file and pre-process
filtered_dataset = pre_process("tripadvisor_co_uk-travel_restaurant_reviews_sample.xlsx")

filtered_dataset["actual_sentiment"] = filtered_dataset.rating.apply(actual_sentiment_swn)    
    
# Predict sentiment score for each of the normalized review texts
filtered_dataset["predicted_sentiment"] = filtered_dataset.normalized_review_text.apply(predict_sentiment_swn)
