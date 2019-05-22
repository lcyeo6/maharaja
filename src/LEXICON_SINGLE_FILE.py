"""

@description:
    
    Lexicon-based Sentiment Analysis

@author: 
    
    Wan Chee Tin
    Yeo Lin Chung
    
"""

import pandas as pd
import datetime
import matplotlib.pyplot as plt
from preprocess import pre_process
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.corpus import wordnet
from nltk.corpus import sentiwordnet

def pre_process_MPQA(filename):
    
    MPQA_Lexicon = pd.read_csv(filename, header = None, names = ["Subjectivity", "Word", "Polarity Score"])
    
    # Construct a Python dictionary to hold the lexicon
    MPQA = {}
    
    # Store each lexicon in a dictionary, e.g. {"Word": ("Subjectivity", "Polarity Score")}
    for index, row in MPQA_Lexicon.iterrows():
        MPQA[row[1]] = (row[0], int(row[2]))
        
    return MPQA

def actual_sentiment_mpqa(data):
    
    # Convert string-format star rating into integer-format
    for i in data:
        return int(i[0])

def predict_sentiment_mpqa(data):
    
    # Counter for weak subjectivity word
    weak_frequency = 0
    
    # Counter for strong subjectivity word
    strong_frequency = 0
    
    # Compute the sentence's sentiment score by total up the word's polarity score
    sentence_score = 0
    for i in data:
        if i[0] in MPQA:
            sentence_score = sentence_score + MPQA[i[0]][1]
            if MPQA[i[0]][0] == 'weaksubj':
                 weak_frequency += 1
            elif MPQA[i[0]][0] == 'strongsubj':
                 strong_frequency += 1
    
    # Strong negative
    if (sentence_score < 0) and (weak_frequency < strong_frequency):
        return 1
    
    # Weak negative
    elif (sentence_score < 0) and (weak_frequency >= strong_frequency):
        return 2
    
    # Strong positive
    elif (sentence_score > 0) and (weak_frequency < strong_frequency):
        return 5
    
    # Weak positive
    elif (sentence_score > 0) and (weak_frequency >= strong_frequency):
        return 4
    
    # Neutral
    else:
        return 3
    
def actual_sentiment_combined(data):
    
    # Convert star-rating into respective sentiments
    for i in data:
        if int(i[0]) == 4 or int(i[0]) == 5:
            return "positive"
        elif int(i[0]) == 1 or int(i[0]) == 2:
            return "negative"
        else:
            return "neutral"

def predict_sentiment_combined(data):
    
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

t1 = datetime.datetime.now()

# Read the data from Excel file and pre-process
filtered_dataset = pre_process("tripadvisor_co_uk-travel_restaurant_reviews_sample.xlsx")

# Read the MPQA lexicon file from CSV file
MPQA = pre_process_MPQA("MPQA/MPQA_Lexicon.csv")
    

"--------- RUN THIS FOR - MPQA ------------------"
#filtered_dataset["actual_sentiment"] = filtered_dataset.rating.apply(actual_sentiment_mpqa)    
#    
#filtered_dataset["predicted_sentiment"] = filtered_dataset.normalized_review_text.apply(predict_sentiment_mpqa)
"----------------------------------------------------------"


"--------- RUN THIS FOR - MPQA SENTIMENT  ------------------"
filtered_dataset["actual_sentiment"] = filtered_dataset.rating.apply(actual_sentiment_combined)    
    
filtered_dataset["predicted_sentiment"] = filtered_dataset.normalized_review_text.apply(predict_sentiment_combined)
"----------------------------------------------------------"


"--------- RUN THIS FOR - MPQA ------------------"
#filtered_dataset["actual_sentiment"] = filtered_dataset.rating.apply(actual_sentiment_combined)    
#    
#filtered_dataset["predicted_sentiment"] = filtered_dataset.normalized_review_text.apply(predict_sentiment_swn)
"----------------------------------------------------------"


print("Pre-process Time")
print(datetime.datetime.now() - t1)
print()


"--------------------------------------------------------------------------------------------------------------------"

"--------- RUN THIS FOR - STATISTICS  ------------------"
# Summary STATISTICS

t2 = datetime.datetime.now()

actual = []
predicted = []

for index, row in filtered_dataset.iterrows():
    actual.append(row[8])
    predicted.append(row[9])

ACC = accuracy_score(actual, predicted)
CR = classification_report(actual, predicted)
CM = confusion_matrix(actual, predicted)
    
print()    
print("Percentage of Accuracy:")
print(ACC)
print() 
print("Classification Report:")
print(CR)
print("Confusion Matrix:")
print(CM)
print()

print("Processing Time")
print(datetime.datetime.now() - t2)
print()
"----------------------------------------------------------"

"---------RUN THIS FOR - SENTIMENT  ------------------"
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
"----------------------------------------------------------"

"--------------------------------------------------------------------------------------------------------------------"
 
# Bar Chart for Sentiment Percentage

#x = [1, 2, 3]
#y = [percentage_negative, percentage_positive, percentage_neutral]
#
#x_label = ["Negative", "Positive", "Neutral"]    
#
#plt.bar(x, y, tick_label = x_label, width = 0.6, color = ["red", "green", "blue"])  
#
#plt.xlabel("Sentiment")
#plt.ylabel("Percentage")
#
#plt.title("Bar Chart")
#
#plt.show()
    
"--------------------------------------------------------------------------------------------------------------------"


