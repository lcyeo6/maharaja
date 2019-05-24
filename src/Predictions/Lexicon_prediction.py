"""

@description:
    
    Lexicon-based Sentiment Analysis

@author: 
    
    Wan Chee Tin
    Yeo Lin Chung
    
"""

import pandas as pd
import datetime
from preprocess import pre_process
import preprocess_user_input

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
    

"--------------------------------------------------------------------------------------------------------------------"

t1 = datetime.datetime.now()

# Read the data from Excel file and pre-process
filtered_dataset = pre_process("tripadvisor_co_uk-travel_restaurant_reviews_sample.xlsx")

# Read the MPQA lexicon file from CSV file
MPQA = pre_process_MPQA("MPQA/MPQA_Lexicon.csv")
    
print("Pre-process Time")
print(datetime.datetime.now() - t1)
print()


"--------------------------------------------------------------------------------------------------------------------"

user_input = input("Enter your review: ")
processed = preprocess_user_input.pre_process(user_input)
print(predict_sentiment_combined(processed))
 



