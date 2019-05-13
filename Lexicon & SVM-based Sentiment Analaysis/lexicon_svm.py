"""

@description:
    
    Lexicon & SVM-based Sentiment Analysis

@author: 
    
    Wan Chee Tin
    Yeo Lin Chung
    
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import KFold

def read_excel(filename):
    
    dataset = pd.read_excel(filename, sheet_name = "Sheet1")
    
    return dataset

def select_data(data):
    select_data = pd.Series([])
    
    for index, row in data.iterrows():
        if (row[8] == 3) or (row[8] == row[9]):
            select_data[index] = 1
        elif row[8] != row[9]:
            select_data[index] = 0
    
    data.insert(10, "one_or_zero", select_data)
    
    data = data[data.one_or_zero != 0]
    return data

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
    
	# Appending all the score into it's own list for averaging the score at the end.
    precision_micro.append(precision_score(y_test,y_pred,average='micro',labels=np.unique(y_pred)))
    recall_micro.append(recall_score(y_test,y_pred,average='micro'))
    f1_micro.append(f1_score(y_test,y_pred,average='micro',labels=np.unique(y_pred)))
    
    precision_macro.append(precision_score(y_test,y_pred,average='macro',labels=np.unique(y_pred)))
    recall_macro.append(recall_score(y_test,y_pred,average='macro'))
    f1_micro.append(f1_score(y_test,y_pred,average='macro',labels=np.unique(y_pred)))
    
    precision_weight.append(precision_score(y_test,y_pred,average='weighted',labels=np.unique(y_pred)))
    recall_weight.append(recall_score(y_test,y_pred,average='weighted'))
    f1_weight.append(f1_score(y_test,y_pred,average='weighted',labels=np.unique(y_pred)))
    return accuracy_train, accuracy_test, precision_micro, recall_micro, f1_micro, precision_macro, recall_macro, f1_macro, precision_weight, recall_weight, f1_weight


def identify_token(text):
#   Return it's text back, as requested as a token
    return text    

"-----------------------------------------------------------------"

# Read the data from Excel file
dataset = read_excel("MPQA_Dataset.xlsx")

# Filter data
filtered_data = select_data(dataset)

"-----------------------------------------------------------------"

# Storing both review texts and star ratings in repective arrays
review_text = []
star_rating = []
for index, row in filtered_data.iterrows():
    review_text.append(row[7])
    star_rating.append(row[8])

# Vectorize review text into unigram, bigram and evaluates into a term document matrix of TF-IDF features
tfidf_vectorizer = TfidfVectorizer(tokenizer=identify_token, ngram_range = (1, 1), lowercase=False)

# Construct vocabulary and inverse document frequency from all the review texts
# Then, transform each review text into a tf-idf weighted document term matrix
vectorized_data = tfidf_vectorizer.fit_transform(review_text)


# LinearSVC
#svc_1 = LinearSVC()

# Normal SVC
svc_1 = SVC(kernel='linear')

# List to store 10-fold of the scores
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
n=0

# Split dataset into training and testing, using 10-fold classification
kfold = KFold(10, True, 1)
for train_index, test_index in kfold.split(vectorized_data, star_rating):
	# Split the data to train and test for each fold
    n +=1
    print(n)
    t3 = datetime.datetime.now()
    X_train, X_test = vectorized_data[train_index], vectorized_data[test_index]
    y_train = [star_rating[i] for i in train_index]
    y_test = [star_rating[i] for i in test_index]
	
	# Train and test the data
    accuracy_train, accuracy_test, precision_micro, recall_micro, f1_micro, precision_macro, recall_macro, f1_macro, precision_weight, recall_weight, f1_weight = train_and_evaluate(svc_1, X_train, X_test, y_train, y_test, accuracy_train, accuracy_test, precision_micro, recall_micro, f1_micro, precision_macro, recall_macro, f1_macro, precision_weight, recall_weight, f1_weight)
    
    print("Train and test time")
    print(datetime.datetime.now() - t3)

# Print out the average scores 
print("accuracy_train: {}".format(np.mean(accuracy_train)))
print("accuracy_test: {}".format(np.mean(accuracy_test)))
print("precision micro: {}".format(np.mean(precision_micro)))
print("recall micro: {}".format(np.mean(recall_micro)))
print("f1 micro: {}".format(np.mean(f1_micro)))
print("precision macro: {}".format(np.mean(precision_macro)))
print("recall macro: {}".format(np.mean(recall_macro)))
print("f1 macro: {}".format(np.mean(f1_macro)))
print("precision weight: {}".format(np.mean(precision_weight)))
print("recall weight: {}".format(np.mean(recall_weight)))
print("f1 weight: {}".format(np.mean(f1_weight)))




