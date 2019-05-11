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
from sklearn.metrics import accuracy_score

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
    
    # Convert string-format star rating into integer-format
    for i in data:
        return int(i[0])  

def predict_sentiment(data):
    
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
        
filtered_dataset["actual_sentiment"] = filtered_dataset.rating.apply(actual_sentiment)    
    
# Predict sentiment score for each of the normalized review texts
filtered_dataset["predicted_sentiment"] = filtered_dataset.normalized_review_text.apply(predict_sentiment)

filtered_dataset.to_excel("MPQA_Dataset.xlsx", index = False)

"--------------------------------------------------------------------------------------------------------------------"

# Summary statistics

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
    
accuracy = accuracy_score(actual, predicted) * 100
    
percentage_one_star = (one_star/total_data) * 100
percentage_two_star = (two_star/total_data) * 100
percentage_three_star = (three_star/total_data) * 100
percentage_four_star = (four_star/total_data) * 100
percentage_five_star = (five_star/total_data) * 100
    
print()    
print("Percentage of Accuracy: %.1f%%" % accuracy)
print("Percentage of Correctly Estimated 1 Star: %.1f%%" % percentage_one_star)
print("Percentage of Correctly Estimated 2 Star: %.1f%%" % percentage_two_star)
print("Percentage of Correctly Estimated 3 Star: %.1f%%" % percentage_three_star)    
print("Percentage of Correctly Estimated 4 Star: %.1f%%" % percentage_four_star) 
print("Percentage of Correctly Estimated 5 Star: %.1f%%" % percentage_five_star) 
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


