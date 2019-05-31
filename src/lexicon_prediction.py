"""

@description:
    
    Lexicon-based Prediction for User Interface

@author: 
    
    Wan Chee Tin
    Yeo Lin Chung

@References:
    https://github.com/stepthom/lexicon-sentiment-analysis
    
"""

import pandas as pd
import preprocess_user_input

def predict_sentiment_combined(MPQA, data):
    
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

def predict(user_input):
    # Read the MPQA lexicon file from CSV file
    MPQA_Lexicon = pd.read_csv("MPQA/MPQA_Lexicon.csv", header = None, names = ["Subjectivity", "Word", "Polarity Score"])
    
    # Construct a Python dictionary to hold the lexicon
    MPQA = {}
    
    # Store each lexicon in a dictionary, e.g. {"Word": ("Subjectivity", "Polarity Score")}
    for index, row in MPQA_Lexicon.iterrows():
        MPQA[row[1]] = (row[0], int(row[2]))
        
    # User input from user interface
    processed = preprocess_user_input.pre_process(user_input)

    # Predicted result from dictionary
    predicted = predict_sentiment_combined(MPQA, processed)

    return predicted


