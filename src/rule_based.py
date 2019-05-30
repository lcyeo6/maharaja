"""

@description:
    
    Sentiment Analysis on MPQA & SentiWordNet

@author: 
    
    Wan Chee Tin
    Yeo Lin Chung
    
"""

import pandas as pd
import datetime

from preprocess import pre_process
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.corpus import wordnet, sentiwordnet

def pre_process_MPQA(filename):
    
    MPQA_Lexicon = pd.read_csv(filename, header = None, names = ["Subjectivity", "Word", "Polarity Score"])
    
    # Construct a Python dictionary to hold the lexicon
    MPQA = {}
    
    # Store each lexicon in a dictionary, e.g. {"Word": ("Subjectivity", "Polarity Score")}
    for index, row in MPQA_Lexicon.iterrows():
        MPQA[row[1]] = (row[0], int(row[2]))
        
    return MPQA
    
def actual_sentiment_combined(data):
    
    # Convert star-rating into respective sentiments
    for i in data:
        if int(i[0]) == 4 or int(i[0]) == 5:
            return "positive"
        elif int(i[0]) == 1 or int(i[0]) == 2:
            return "negative"
        else:
            return "neutral"

def predict_sentiment_mpqa(data):
    
    # Compute the sentence's sentiment score by total up the word's polarity score
    sentence_score = 0
    
    for i in data:
        if i[0] in MPQA:
            sentence_score = sentence_score + MPQA[i[0]][1]
    
    if sentence_score > 0:
        return "positive"
    elif sentence_score < 0:
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
    
"--------------------------------------------------------------------------------------------------------------------"
"--------------------------------------------------------------------------------------------------------------------"



"---------------------------------------------------- PRE-PROCESS ----------------------------------------------------"
#t1 = datetime.datetime.now()
#
## Read the data from Excel file and pre-process
#filtered_dataset = pre_process("tripadvisor_co_uk-travel_restaurant_reviews_sample.xlsx")
#
# Read the MPQA lexicon file from CSV file
MPQA = pre_process_MPQA("MPQA/MPQA_Lexicon.csv")

#print("Pre-process Time")
#print(datetime.datetime.now() - t1)
#print()

"--------------------------------------------------------------------------------------------------------------------"
    


"------------------------------------------------------ MPQA ------------------------------------------------------"
#filtered_dataset["actual_sentiment"] = filtered_dataset.rating.apply(actual_sentiment_mpqa)    
#    
#filtered_dataset["predicted_sentiment"] = filtered_dataset.normalized_review_text.apply(predict_sentiment_mpqa)
"------------------------------------------------------------------------------------------------------------------"



"------------------------------------------------- MPQA SENTIMENT -------------------------------------------------"
#filtered_dataset["actual_sentiment"] = filtered_dataset.rating.apply(actual_sentiment_combined)    
#    
#filtered_dataset["predicted_sentiment"] = filtered_dataset.normalized_review_text.apply(predict_sentiment_combined)
"------------------------------------------------------------------------------------------------------------------"



"--------------------------------------------- SENTIWORDNET SENTIMENT ---------------------------------------------"
#filtered_dataset["actual_sentiment"] = filtered_dataset.rating.apply(actual_sentiment_combined)    
#    
#filtered_dataset["predicted_sentiment"] = filtered_dataset.normalized_review_text.apply(predict_sentiment_swn)
"------------------------------------------------------------------------------------------------------------------"




"-------------------------------------------------------------------------------------------------------------------"



"--------------------------------------------- STATISTIC - STAR RATING ---------------------------------------------"
# Summary STATISTICS
#actual = []
#predicted = []
#
#for index, row in filtered_dataset.iterrows():
#    actual.append(row[8])
#    predicted.append(row[9])
#
#ACC = accuracy_score(actual, predicted)
#CR = classification_report(actual, predicted)
#CM = confusion_matrix(actual, predicted)
#    
#print()    
#print("Percentage of Accuracy:")
#print(ACC)
#print() 
#print("Classification Report:")
#print(CR)
#print("Confusion Matrix:")
#print(CM)
#print()
"-------------------------------------------------------------------------------------------------------------------"



"---------------------------------------------- STATISTIC - SENTIMENT ----------------------------------------------"
## Summary SENTIMENTS
#actual = []
#predicted = []
#
#for index, row in filtered_dataset.iterrows():
#    actual.append(row[8])
#    predicted.append(row[9])
#    
#ACC = accuracy_score(actual, predicted) * 100
#CR = classification_report(actual, predicted)
#CM = confusion_matrix(actual, predicted)
#
#print()    
#print("Percentage of Accuracy: %.1f%%" % ACC)
#print()
#print("Classification Report")
#print(CR)
#print("Confusion Matrix")
#print(CM)
#print()
"-------------------------------------------------------------------------------------------------------------------"
