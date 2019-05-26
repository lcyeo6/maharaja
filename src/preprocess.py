"""

@description:
    
    Pre-process an excel file

@author: 
    
    Wan Chee Tin
    Yeo Lin Chung
    
@update:

    1. (06/03/2019) 2 rows of data were removed from the Excel file because both consist of date in 'rating' column.
    2. (12/03/2019) Added comments for each function.
    
"""

import pandas as pd
import re
import datetime
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet

def wordnet_tag(tag):
    
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('R'):
        return wordnet.ADV
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('V'):
        return wordnet.VERB
    return None

def pre_process(filename):
    
    dataset = pd.read_excel(filename, sheet_name = "Sheet1")
    
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
        tokens = word_tokenize(letters)
        
        # Convert tokenized words into tokenized lower-case words
        small_letters = []
        for i in tokens:
            small_letters.append(i.lower())
        
        # Remove stop words 
        filtered = list(filter(lambda l: l not in stop_words, small_letters))
        
        # Parts of Speech Tagging
        tagged_filtered = pos_tag(filtered)
        
        # Lemmatize using WordNet's built-in function
        lemmatized_words = []
        for word, tag in tagged_filtered:
            wn_tag = wordnet_tag(tag)
            if wn_tag not in (wordnet.ADJ, wordnet.ADV, wordnet.NOUN):
                continue
            lemmatized_word = lemmatizer.lemmatize(word, pos = wn_tag)
            if not lemmatized_word:
                continue
            else:
                lemmatized_words.append((lemmatized_word, wn_tag))
            
        return lemmatized_words
    
    # Apply text_normalization function
    dataset["normalized_review_text"] = dataset.review_text.apply(text_normalization)
    
    return dataset


# Read from Excel & Data cleaning & pre-processing
#filtered_dataset = pre_process("tripadvisor_co_uk-travel_restaurant_reviews_sample.xlsx")
#
#filtered_dataset.to_excel("preprocessed.xlsx")



# Read the data from Excel file
#dataset = read_excel("tripadvisor_co_uk-travel_restaurant_reviews_sample.xlsx")

#def read_excel(filename):
#    
#    dataset = pd.read_excel(filename, sheet_name = "Sheet1")
#    
#    return dataset

#def ngrams(data):
#    
#    # Generate sequences of normalized words beginning from distinct elements of the list of normalized words
#    # The zip function takes the sequences as a list of inputs
#    # bigram = A sequence of two adjacent words
#    bigram = []
#    for i in list(zip(data, data[1:])):
#        bigram.append(' '.join(i))
#    
#    # Generate sequences of normalized words beginning from distinct elements of the list of normalized words
#    # The zip function takes the sequences as a list of inputs
#    # trigram = A sequence of three adjacent words
#    trigram = []
#    for j in list(zip(data, data[1:], data[2:])):
#        trigram.append(' '.join(j))
#    
#    return bigram + trigram