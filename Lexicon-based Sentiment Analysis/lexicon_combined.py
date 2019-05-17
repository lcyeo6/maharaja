"""

@description:
    
    Lexicon-based Sentiment Analysis

@author: 
    
    Wan Chee Tin
    Yeo Lin Chung
    
"""

import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def read_excel(filename):
    
    dataset = pd.read_excel(filename, sheet_name = "Sheet1")
    
    return dataset

def pre_process(dataset):
    
    # Remove unwanted columns
    dataset = dataset.drop(["uniq_id", "url", "restaurant_id", "restaurant_location", "category", "review_date", "author", "author_url", "location", "visited_on"], axis = 1)
    
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

# Read the data from Excel file
dataset = read_excel("tripadvisor_co_uk-travel_restaurant_reviews_sample.xlsx")

# Data cleaning & pre-processing
filtered_dataset = pre_process(dataset)

"--------------------------------------------------------------------------------------------------------------------"

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


