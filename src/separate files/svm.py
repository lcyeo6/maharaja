"""
@description:
    
    SVM-based Sentiment Analysis
@author: 
    
    Wan Chee Tin
    Yeo Lin Chung
    
"""

import numpy as np
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import KFold

def identify_token(text):
    
    # Return it's text back, as requested as a token
    return text

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

"--------------------------------------------------------------------------------------------------------------------"
def run_svm(filtered_dataset):
    # Storing both review texts and star ratings in repective arrays
    review_text = []
    star_rating = []
    for index, row in filtered_dataset.iterrows():
        review_text.append(row[7])
        star_rating.append(int(row[3][0]))
    
    # Vectorize review text into unigram, bigram and evaluates into a term document matrix of TF-IDF features
    false_ans = True
    while false_ans == True:
        user_input = int(input("Choose either 1: Unigram || 2: Unigram & Bigram"))
        if user_input == 1 or user_input == 2:
            false_ans = False
            break
    tfidf_vectorizer = TfidfVectorizer(tokenizer = identify_token, ngram_range = (1, user_input), lowercase = False)
    
    t2 = datetime.datetime.now()
    
    # Construct vocabulary and inverse document frequency from all the review texts
    # Then, transform each review text into a tf-idf weighted document term matrix
    vectorized_data = tfidf_vectorizer.fit_transform(review_text)
    
    print("Vectorizing Time")
    print(datetime.datetime.now() - t2)
    print()
    
    "--------------------------------------------------------------------------------------------------------------------"  
      
    false_ans_2 = True
    while false_ans_2 == True:
        user_input_2 = int(input("Choose 1: Normal SVC || 2: Linear SVC || 3: RBF"))
        if user_input_2 == 1 or user_input_2 == 2 or user_input_2 == 3:
            false_ans_2 = False
            break
    if user_input_2 == 1:
        # Normal SVC
        svc = SVC(kernel = 'linear')
    elif user_input_2 == 2:
        # LinearSVC
        svc = LinearSVC()
    else:
        svc = SVC(kernel='rbf')
    
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
    
    # Split dataset into training and testing using 10-fold classification
    kfold = KFold(10, True, 1)
    
    for train_index, test_index in kfold.split(vectorized_data, star_rating):
        n += 1
        print(n)
        t3 = datetime.datetime.now()
        X_train, X_test = vectorized_data[train_index], vectorized_data[test_index]
        y_train = [star_rating[i] for i in train_index]
        y_test = [star_rating[i] for i in test_index]
    	
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

