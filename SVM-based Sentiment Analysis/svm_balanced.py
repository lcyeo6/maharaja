# -*- coding: utf-8 -*-
"""
@description:
    
    SVM-based Sentiment Analysis
@author: 
    
    Wan Chee Tin
    Yeo Lin Chung
    
"""

import pandas as pd
import numpy as np
import re
import datetime
from collections import Counter
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import KFold

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

def class_equity(a, b):
    # Count the frequency of each category
    occurrence = Counter(b)
    
    # The least common category will be the minimum equity
    maximum = occurrence.most_common()[-1][1]

    # Serve as counter
    total = dict()
    for category in occurrence.keys():
        total[category] = 0

    equalized_a = []
    equalized_b = []
	
	# Number of words in the least number of reviews 
    no_of_words = {"50":0, "100":0, "150":0, "200":0, "250":0, "300":0}
	
	# Counter to keep track of the number of words of the current data
    counter_1 = {"50":0, "100":0, "150":0, "200":0, "250":0, "300":0}
    counter_2 = {"50":0, "100":0, "150":0, "200":0, "250":0, "300":0}
    counter_3 = {"50":0, "100":0, "150":0, "200":0, "250":0, "300":0}
    counter_4 = {"50":0, "100":0, "150":0, "200":0, "250":0, "300":0}
    counter_5 = {"50":0, "100":0, "150":0, "200":0, "250":0, "300":0}
	
	# To check the no of words in 1-star rating data
    for i, e in enumerate(b):
        if e == 1:
            if len(a[i]) <= 50:
                no_of_words["50"] += 1
            elif len(a[i]) > 50 and len(a[i]) <= 100:
                no_of_words["100"] += 1
            elif len(a[i]) > 100 and len(a[i]) <= 150:
                no_of_words["150"] += 1
            elif len(a[i]) > 150 and len(a[i]) <= 200:
                no_of_words["200"] += 1
            elif len(a[i]) > 200 and len(a[i]) <= 250:
                no_of_words["250"] += 1
            else:
                no_of_words["300"] += 1         
        
    def equalizer(counter):
	# Balance the dataset by removing over-represented samples from two arrays
        if total[element] < maximum:
            if len(a[index]) <= 50:
                if counter["50"] < no_of_words["50"]:
                    counter["50"] += 1
                    equalized_a.append(a[index])
                    equalized_b.append(element)
                    total[element] += 1
            elif len(a[index]) > 50 and len(a[index]) <= 100:
                if counter["100"] < no_of_words["100"]:
                    counter["100"] += 1
                    equalized_a.append(a[index])
                    equalized_b.append(element)
                    total[element] += 1
            elif len(a[index]) > 100 and len(a[index]) <= 150:
                if counter["150"] < no_of_words["150"]:
                    counter["150"] += 1
                    equalized_a.append(a[index])
                    equalized_b.append(element)
                    total[element] += 1
            elif len(a[index]) > 150 and len(a[index]) <= 200:
                if counter["200"] < no_of_words["200"]:
                    counter["200"] += 1
                    equalized_a.append(a[index])
                    equalized_b.append(element)
                    total[element] += 1
            elif len(a[index]) > 200 and len(a[index]) <= 250:
                if counter["250"] < no_of_words["250"]:
                    counter["250"] += 1
                    equalized_a.append(a[index])
                    equalized_b.append(element)
                    total[element] += 1
            elif counter["300"] < no_of_words["300"]:
                    counter["300"] += 1
                    equalized_a.append(a[index])
                    equalized_b.append(element)
                    total[element] += 1
                    
	# Balance data for each star rating
    for index, element in enumerate(b):
        if element == 1:
            equalizer(counter_1)
        elif element == 2:
            equalizer(counter_2)
        elif element == 3:
            equalizer(counter_3)
        elif element == 4:
            equalizer(counter_4)
        else:
            equalizer(counter_5)

    return equalized_a, equalized_b

def identify_token(text):
#   Return it's text back, as requested as a token
    return text

"--------------------------------------------------------------------------------------------------------------------"

# Read the data from Excel file
dataset = read_excel("tripadvisor_co_uk-travel_restaurant_reviews_sample.xlsx")
t1 = datetime.datetime.now()
# Data cleaning & pre-processing
filtered_dataset = pre_process(dataset)

"--------------------------------------------------------------------------------------------------------------------"

# Storing both normalized review texts and star ratings in repective arrays
review_text = []
star_rating = []
for index, row in filtered_dataset.iterrows():
    review_text.append(row[7])
    star_rating.append(int(row[3][0]))

# Balance the data we are going to put into training and testing
equalized_review_text, equalized_star_rating = class_equity(review_text, star_rating)

print("Pre-process Time")
print(datetime.datetime.now() - t1)
print()

# Vectorize review text into unigram, bigram and evaluates into a term document matrix of TF-IDF features
tfidf_vectorizer = TfidfVectorizer(tokenizer = identify_token, ngram_range = (1, 1), lowercase = False)
t2 = datetime.datetime.now()

# Construct vocabulary and inverse document frequency from all the review texts
# Then, transform each review text into a tf-idf weighted document term matrix
vectorized_data = tfidf_vectorizer.fit_transform(equalized_review_text)

print("Vectorizing Time")
print(datetime.datetime.now() - t2)
print()

"--------------------------------------------------------------------------------------------------------------------"

def train_and_evaluate(clf, X_train, X_test, y_train, y_test, accuracy_train, accuracy_test, precision_micro, recall_micro, f1_micro, precision_macro, recall_macro, f1_macro, precision_weight, recall_weight, f1_weight):
    # Perform training on the training set
    clf.fit(X_train, y_train)
    
#    print ("Accuracy on training set:")
#    print (clf.score(X_train, y_train))
    accuracy_train.append(clf.score(X_train, y_train))
#    print ("Accuracy on testing set:")
#    print (clf.score(X_test, y_test))
    accuracy_test.append(clf.score(X_test, y_test))
    
	# Predicting the training data used on the test data
    y_pred = clf.predict(X_test)
    
    print ("Classification Report:")
    print (metrics.classification_report(y_test, y_pred))
    print ("Confusion Matrix:")
    print (metrics.confusion_matrix(y_test, y_pred))
    print()
    
	# Appending all the scores into it's own list for averaging the score at the end
    precision_micro.append(precision_score(y_test, y_pred,average = 'micro', labels = np.unique(y_pred)))
    recall_micro.append(recall_score(y_test, y_pred, average = 'micro'))
    f1_micro.append(f1_score(y_test, y_pred, average = 'micro', labels = np.unique(y_pred)))
    
    precision_macro.append(precision_score(y_test, y_pred, average = 'macro', labels = np.unique(y_pred)))
    recall_macro.append(recall_score(y_test, y_pred, average = 'macro'))
    f1_macro.append(f1_score(y_test, y_pred, average = 'macro', labels = np.unique(y_pred)))
    
    precision_weight.append(precision_score(y_test,y_pred,average = 'weighted', labels = np.unique(y_pred)))
    recall_weight.append(recall_score(y_test, y_pred, average = 'weighted'))
    f1_weight.append(f1_score(y_test, y_pred, average = 'weighted', labels = np.unique(y_pred)))
    
# LinearSVC
#svc = LinearSVC()

# Normal SVC
svc = SVC(kernel = 'linear')
    
# RBF
#svc = SVC(kernel='rbf')

accuracy_train = []
accuracy_test = []

precision_micro = []
recall_micro = []
f1_micro = []

precision_macro = []
recall_macro = []
f1_macro = []

precision_weight = []
recall_weight = []
f1_weight = []

# To count which fold the program is currently at
n = 0

# Split dataset into training and testing, using 10-fold classification
kfold = KFold(10, True, 1)
for train_index, test_index in kfold.split(vectorized_data, equalized_star_rating):
    n += 1
    print(n)
    t3 = datetime.datetime.now()
    X_train, X_test = vectorized_data[train_index], vectorized_data[test_index]
    y_train = [equalized_star_rating[i] for i in train_index]
    y_test = [equalized_star_rating[i] for i in test_index]
	
	# Train and test the data
    train_and_evaluate(svc, X_train, X_test, y_train, y_test, accuracy_train, accuracy_test, precision_micro, recall_micro, f1_micro, precision_macro, recall_macro, f1_macro, precision_weight, recall_weight, f1_weight)
    
    print("Train and Test Time")
    print(datetime.datetime.now() - t3)
    print()
    print("------------------------------------------")
    
# Print out the average scores
print("accuracy train:   {}".format(np.mean(accuracy_train)))
print("accuracy test:    {}".format(np.mean(accuracy_test)))
print()
print("precision micro:  {}".format(np.mean(precision_micro)))
print("recall micro:     {}".format(np.mean(recall_micro)))
print("f1 micro:         {}".format(np.mean(f1_micro)))
print()
print("precision macro:  {}".format(np.mean(precision_macro)))
print("recall macro:     {}".format(np.mean(recall_macro)))
print("f1 macro:         {}".format(np.mean(f1_macro)))
print()
print("precision weight: {}".format(np.mean(precision_weight)))
print("recall weight:    {}".format(np.mean(recall_weight)))
print("f1 weight:        {}".format(np.mean(f1_weight)))