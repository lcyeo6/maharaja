"""

@description:
    
    Lexicon-based Sentiment Analysis

@author: 
    
    Wan Chee Tin
    Yeo Lin Chung
    
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

"--------------------------------------------------------------------------------------------------------------------"
def run_lexicon(filtered_dataset):
    # Read the MPQA lexicon file from CSV file
    MPQA_Lexicon = pd.read_csv("MPQA/MPQA_Lexicon.csv", header = None, names = ["Subjectivity", "Word", "Polarity Score"])
    
    # Construct a Python dictionary to hold the lexicon
    MPQA = {}
    
    # Store each lexicon in a dictionary, e.g. {"Word": ("Subjectivity", "Polarity Score")}
    for index, row in MPQA_Lexicon.iterrows():
        MPQA[row[1]] = (row[0], int(row[2]))
        
    def actual_sentiment(data):
        
        # Convert star-rating into respective sentiments
        for i in data:
            if int(i[0]) == 4 or int(i[0]) == 5:
                return "positive"
            elif int(i[0]) == 1 or int(i[0]) == 2:
                return "negative"
            else:
                return "neutral"
    
    def predict_sentiment(data):
        
        # Compute the sentence's sentiment score by total up the word's polarity score
        sentence_score = 0
        
        for i in data:
            if i in MPQA:
                sentence_score = sentence_score + MPQA[i][1]
        
        if sentence_score > 0:
            return "positive"
        elif sentence_score < 0:
            return "negative"
        else:
            return "neutral"
        
    filtered_dataset["actual_sentiment"] = filtered_dataset.rating.apply(actual_sentiment)  
    
    # Predict sentiment score for each of the normalized review texts
    filtered_dataset["predicted_sentiment"] = filtered_dataset.normalized_review_text.apply(predict_sentiment)
    
    "--------------------------------------------------------------------------------------------------------------------"
    
    # Summary statistics
    
    actual = []
    predicted = []
    
    for index, row in filtered_dataset.iterrows():
        actual.append(row[8])
        predicted.append(row[9])
        
    ACC = accuracy_score(actual, predicted) * 100
    CR = classification_report(actual, predicted)
    CM = confusion_matrix(actual, predicted)
    
    print()    
    print("Percentage of Accuracy: %.1f%%" % ACC)
    print()
    print("Classification Report")
    print(CR)
    print("Confusion Matrix")
    print(CM)
    print()
    
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
    

