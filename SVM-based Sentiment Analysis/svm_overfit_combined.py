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
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import KFold

# Required to download conda install -c conda-forge imbalanced-learn
from imblearn.over_sampling import ADASYN

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


def identify_token(text):
    
    # Return it's text back, as requested as a token
    return text

def actual_sentiment(data):
    
    # Convert star-rating into respective sentiments
    for i in data:
        if int(i[0]) == 4 or int(i[0]) == 5:
            return "positive"
        elif int(i[0]) == 1 or int(i[0]) == 2:
            return "negative"
        else:
            return "neutral"

def train_and_evaluate(clf, X_train, X_test, y_train, y_test, accuracy_train, accuracy_test, precision_micro, recall_micro, f1_micro, precision_macro, recall_macro, f1_macro, precision_weight, recall_weight, f1_weight):
	
    # Overfit the training data using ADASYN
    sm = ADASYN()
    overfit_X, overfit_y = sm.fit_sample(X_train, y_train)
	
    # Perform training on the training set
    clf.fit(overfit_X, overfit_y)
    
#    print ("Accuracy on training set:")
#    print (clf.score(X_train, y_train))
    accuracy_train.append(clf.score(overfit_X, overfit_y))
#    print ("Accuracy on testing set:")
#    print (clf.score(X_test, y_test))
    accuracy_test.append(clf.score(X_test, y_test))
    
	# Predicting the training data used on the test data
    y_pred = clf.predict(X_test)
    
    print ("Classification Report:")
    print (classification_report(y_test, y_pred))
    print ("Confusion Matrix:")
    print (confusion_matrix(y_test, y_pred))
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
    
    return accuracy_train, accuracy_test, precision_micro, recall_micro, f1_micro, precision_macro, recall_macro, f1_macro, precision_weight, recall_weight, f1_weight

"--------------------------------------------------------------------------------------------------------------------"

# Read the data from Excel file
dataset = read_excel("tripadvisor_co_uk-travel_restaurant_reviews_sample.xlsx")

t1 = datetime.datetime.now()

# Data cleaning & pre-processing
filtered_dataset = pre_process(dataset)

filtered_dataset["actual_sentiment"] = filtered_dataset.rating.apply(actual_sentiment)  

filtered_dataset = filtered_dataset[filtered_dataset.actual_sentiment != "neutral"]  

print("Pre-process Time")
print(datetime.datetime.now() - t1)
print()

"--------------------------------------------------------------------------------------------------------------------"

# Storing both review texts and star ratings in repective arrays
review_text = []
sentiment = []
for index, row in filtered_dataset.iterrows():
    review_text.append(row[7])
    sentiment.append(row[8])
 
# Vectorize review text into unigram, bigram and evaluates into a term document matrix of TF-IDF features
tfidf_vectorizer = TfidfVectorizer(tokenizer = identify_token, ngram_range = (1, 2), lowercase = False)

t2 = datetime.datetime.now()

# Construct vocabulary and inverse document frequency from all the review texts
# Then, transform each review text into a tf-idf weighted document term matrix
vectorized_data = tfidf_vectorizer.fit_transform(review_text)

print("Vectorizing Time")
print(datetime.datetime.now() - t2)
print()

"--------------------------------------------------------------------------------------------------------------------"

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
for train_index, test_index in kfold.split(vectorized_data, sentiment):
    n += 1
    print(n)
    t3 = datetime.datetime.now()
    X_train, X_test = vectorized_data[train_index], vectorized_data[test_index]
    y_train = [sentiment[i] for i in train_index]
    y_test = [sentiment[i] for i in test_index]
	
	# Train and test the data
    accuracy_train, accuracy_test, precision_micro, recall_micro, f1_micro, precision_macro, recall_macro, f1_macro, precision_weight, recall_weight, f1_weight = train_and_evaluate(svc, X_train, X_test, y_train, y_test, accuracy_train, accuracy_test, precision_micro, recall_micro, f1_micro, precision_macro, recall_macro, f1_macro, precision_weight, recall_weight, f1_weight)
	
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