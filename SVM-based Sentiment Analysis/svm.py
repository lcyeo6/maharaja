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
    print(occurrence)
    print(maximum)

    # Serve as counter
    total = dict()
    for category in occurrence.keys():
        total[category] = 0

    equalized_a = []
    equalized_b = []
    
#    print(statistics.median(word_count))
        
    # Balance the dataset by removing over-represented samples from two arrays
    for index, element in enumerate(b):
#        if total[element] < maximum:
        equalized_a.append(a[index])
        equalized_b.append(element)
        total[element] += 1
    return equalized_a, equalized_b

def identify_token(text):
#   Return it's text back, as requested as a token
    return text



"-----------------------------------------------------------------"

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
    
equalized_review_text, equalized_star_rating = class_equity(review_text, star_rating)
print("PRE-process time")
print(datetime.datetime.now() - t1)


# Vectorize review text into unigram, bigram and evaluates into a term document matrix of TF-IDF features
tfidf_vectorizer = TfidfVectorizer(tokenizer=identify_token, ngram_range = (1, 2), lowercase=False)
t2 = datetime.datetime.now()

# Construct vocabulary and inverse document frequency from all the review texts
# Then, transform each review text into a tf-idf weighted document term matrix
vectorized_data = tfidf_vectorizer.fit_transform(equalized_review_text)
#print(vectorized_data)
print("Vectorizing time")
print(datetime.datetime.now() - t2)

"-----------------------------------------------------------------"

#def evaluate_cross_validation(clf, X, y, K):
#    # create a k-fold croos validation iterator
#    cv = KFold(10, True, 1)
#    # by default the score used is the one returned by score method of the estimator (accuracy)
#    scores = cross_val_score(clf, X, y, cv=cv)
#    print (scores)
#    print (("Mean score: {0:.3f} (+/-{1:.3f})").format(np.mean(scores), sem(scores)))

def train_and_evaluate(clf, X_train, X_test, y_train, y_test, accuracy_train, accuracy_test, precision, recall, f1):
    # Function to perform training on the training set and evaluate the performance on the testing set
    clf.fit(X_train, y_train)
    
#    print ("Accuracy on training set:")
#    print (clf.score(X_train, y_train))
    accuracy_train.append(clf.score(X_train, y_train))
#    print ("Accuracy on testing set:")
#    print (clf.score(X_test, y_test))
    accuracy_test.append(clf.score(X_test, y_test))
    
    y_pred = clf.predict(X_test)
    
#    print ("Classification Report:")
#    print (metrics.classification_report(y_test, y_pred))
    print ("Confusion Matrix:")
    print (metrics.confusion_matrix(y_test, y_pred))
    
    precision.append(precision_score(y_test,y_pred,average='micro',labels=np.unique(y_pred)))
    recall.append(recall_score(y_test,y_pred,average='micro'))
    f1.append(f1_score(y_test,y_pred,average='micro',labels=np.unique(y_pred)))
    

svc_1 = SVC(kernel='linear')
#print (svc_1)

accuracy_train = []
accuracy_test = []
precision = []
recall = []
f1 = []
# Split dataset into training and testing

kfold = KFold(10, True, 1)
for train_index, test_index in kfold.split(vectorized_data, equalized_star_rating):
    t3 = datetime.datetime.now()
    X_train, X_test = vectorized_data[train_index], vectorized_data[test_index]
    y_train = [equalized_star_rating[i] for i in train_index]
    y_test = [equalized_star_rating[i] for i in test_index]
    train_and_evaluate(svc_1, X_train, X_test, y_train, y_test, accuracy_train, accuracy_test, precision, recall, f1)
    print("Train and test time")
    print(datetime.datetime.now() - t3)

print("DONE 10 times")
print("accuracy_train: {}".format(np.mean(accuracy_train)))
print("accuracy_test: {}".format(np.mean(accuracy_test)))
print("precision: {}".format(np.mean(precision)))
print("recall: {}".format(np.mean(recall)))
print("f1: {}".format(np.mean(f1)))



# Evaluate K-fold cross-validation with 10-folds
#evaluate_cross_validation(svc_1, X_train, y_train, 10)


# Perform training on training data and evaluate performance on testing data
#train_and_evaluate(svc_1, X_train, X_test, y_train, y_test)
#print("Train and test time")
#print(datetime.datetime.now() - t3)


## K-fold cross validation
#from sklearn.model_selection import KFold
#kfold = KFold(10, True, 1)
#for train, test in kfold.split(vectorized_data):
#    print('train: %s, test: %s' % (vectorized_data[train], vectorized_data[test]))

#from sklearn.cross_validation import cross_val_score, cross_val_predict
#from sklearn import metrics
## Perform 6-fold cross validation
#scores = cross_val_score(model, df, y, cv=6)
#print('Cross-validated scores:', scores)




##train_X, test_X, train_Y, test_Y = train_test_split(vectorized_data, equalized_star_rating)
