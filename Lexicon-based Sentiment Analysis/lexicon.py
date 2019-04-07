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
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

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

# Read the data from Excel file
dataset = read_excel("tripadvisor_co_uk-travel_restaurant_reviews_sample.xlsx")

# Data cleaning & pre-processing
filtered_dataset = pre_process(dataset)
filtered_dataset["ngrams"] = filtered_dataset.normalized_review_text.apply(ngrams)

"--------------------------------------------------------------------------------------------------------------------"

# Read the MPQA lexicon file from CSV file
MPQA_Lexicon = pd.read_csv("MPQA/MPQA_Lexicon.csv", header = None)
MPQA_Lexicon.columns = ["Word", "Polarity Score"]

# Construct a Python dictionary to hold the lexicon
MPQA = {}

for index, row in MPQA_Lexicon.iterrows():
    MPQA[row[0]] = int(row[1])
    
def lexicon_analysis(data):
    
    sentence_score = 0
    for i in data:
        if i in MPQA:
            sentence_score = sentence_score + MPQA[i]
    
    if (sentence_score < 0):
        return sentence_score, "negative"
    elif (sentence_score > 0):
        return sentence_score, "positive"
    else:
        return sentence_score, "neutral"
    
filtered_dataset["total_sentiment_score"] = filtered_dataset.normalized_review_text.apply(lexicon_analysis)
    
# Summary statistics
total_data = len(filtered_dataset)
total_negative = 0
total_positive = 0
total_neutral = 0

for row in filtered_dataset["total_sentiment_score"]:
    if row[1] == "negative":
        total_negative += 1
    elif row[1] == "positive":
        total_positive += 1
    else:
        total_neutral += 1
    
print()    
print("Percentage of Negative Sentiment: %.1f%%" % ((total_negative/total_data) * 100))
print("Percentage of Positive Sentiment: %.1f%%" % ((total_positive/total_data) * 100))
print("Percentage of Neutral Sentiment: %.1f%%" % ((total_neutral/total_data) * 100))    
    
    
    
    
    
    
    
    