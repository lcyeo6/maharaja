"""

@description:
    
    Lexicon-based Sentiment Analysis

@author: 
    
    Wan Chee Tin
    Yeo Lin Chung
    
"""

import pandas as pd
import preprocess
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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
        if i in MPQA:
            sentence_score = sentence_score + MPQA[i][1]
            if MPQA[i][0] == 'weaksubj':
                 weak_frequency += 1
            elif MPQA[i][0] == 'strongsubj':
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
        if i in MPQA:
            sentence_score = sentence_score + MPQA[i][1]
    
    if sentence_score > 0:
        return "positive"
    elif sentence_score < 0:
        return "negative"
    else:
        return "neutral"

"--------------------------------------------------------------------------------------------------------------------"

# Read the data from Excel file and pre-process
filtered_dataset = preprocess.pre_process("tripadvisor_co_uk-travel_restaurant_reviews_sample.xlsx")

# Read the MPQA lexicon file from CSV file
MPQA_Lexicon = pd.read_csv("MPQA/MPQA_Lexicon.csv", header = None, names = ["Subjectivity", "Word", "Polarity Score"])

# Construct a Python dictionary to hold the lexicon
MPQA = {}

# Store each lexicon in a dictionary, e.g. {"Word": ("Subjectivity", "Polarity Score")}
for index, row in MPQA_Lexicon.iterrows():
    MPQA[row[1]] = (row[0], int(row[2]))
    
    
"---------RUN THIS FOR - STATISTICS  ------------------"
filtered_dataset["actual_sentiment"] = filtered_dataset.rating.apply(actual_sentiment_mpqa)    
    
# Predict sentiment score for each of the normalized review texts
filtered_dataset["predicted_sentiment"] = filtered_dataset.normalized_review_text.apply(predict_sentiment_mpqa)
"----------------------------------------------------------"

"---------RUN THIS FOR - SENTIMENT  ------------------"
#filtered_dataset["actual_sentiment"] = filtered_dataset.rating.apply(actual_sentiment_combined)    
#    
## Predict sentiment score for each of the normalized review texts
#filtered_dataset["predicted_sentiment"] = filtered_dataset.normalized_review_text.apply(predict_sentiment_combined)
"----------------------------------------------------------"

filtered_dataset.to_excel("MPQA_Dataset.xlsx", index = False)

"--------------------------------------------------------------------------------------------------------------------"

"---------RUN THIS FOR - STATISTICS  ------------------"
# Summary STATISTICS

total_data = len(filtered_dataset)
one_star = 0
two_star = 0
three_star = 0
four_star = 0
five_star = 0

actual = []
predicted = []

for index, row in filtered_dataset.iterrows():
    actual.append(row[8])
    predicted.append(row[9])
    
    if row[8] == 1 and row[9] == 1:
        one_star += 1
    if row[8] == 2 and row[9] == 2:
        two_star += 1
    if row[8] == 3 and row[9] == 3:
        three_star += 1
    if row[8] == 4 and row[9] == 4:
        four_star += 1
    if row[8] == 5 and row[9] == 5:
        five_star += 1
    
ACC = accuracy_score(actual, predicted) * 100
CR = classification_report(actual, predicted)
CM = confusion_matrix(actual, predicted)
    
percentage_one_star = (one_star/total_data) * 100
percentage_two_star = (two_star/total_data) * 100
percentage_three_star = (three_star/total_data) * 100
percentage_four_star = (four_star/total_data) * 100
percentage_five_star = (five_star/total_data) * 100
    
print()    
print("Percentage of Accuracy: %.1f%%" % ACC)
print()
print("Classification Report")
print(CR)
print("Confusion Matrix")
print(CM)
print()
print("Percentage of Correctly Estimated 1 Star: %.1f%%" % percentage_one_star)
print("Percentage of Correctly Estimated 2 Star: %.1f%%" % percentage_two_star)
print("Percentage of Correctly Estimated 3 Star: %.1f%%" % percentage_three_star)    
print("Percentage of Correctly Estimated 4 Star: %.1f%%" % percentage_four_star) 
print("Percentage of Correctly Estimated 5 Star: %.1f%%" % percentage_five_star) 
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


