"""
@description:
    
    SVM-based Sentiment Analysis
@author: 
    
    Wan Chee Tin
    Yeo Lin Chung
    
"""

import numpy as np
import pandas as pd
import datetime
import preprocess
from imblearn.over_sampling import ADASYN
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import KFold

def class_equity(a, b):
    
    # Count the frequency of each category
    occurrence = Counter(b)
    
    # The least common category will be the minimum equity
    maximum_star_rating = occurrence.most_common()[-1][0]
    maximum_amount = occurrence.most_common()[-1][1]
    
    # Serve as counter
    total = dict()
    for category in occurrence.keys():
        total[category] = 0

    equalized_a = []
    equalized_b = []
	
	# Number of words in the least number of reviews 
    no_of_words = {"50": 0, "100": 0, "150": 0, "200": 0, "250": 0, "300": 0}
	
	# To check the number of words of the review text with the least amount of star rating
    for i, star_rating in enumerate(b):
        if star_rating == maximum_star_rating:
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
                
    # Counter to keep track of the number of words of the review text with respective star rating
    counter_1 = {"50": 0, "100": 0, "150": 0, "200": 0, "250": 0, "300": 0}
    counter_2 = {"50": 0, "100": 0, "150": 0, "200": 0, "250": 0, "300": 0}
    counter_3 = {"50": 0, "100": 0, "150": 0, "200": 0, "250": 0, "300": 0}
    counter_4 = {"50": 0, "100": 0, "150": 0, "200": 0, "250": 0, "300": 0}
    counter_5 = {"50": 0, "100": 0, "150": 0, "200": 0, "250": 0, "300": 0}
    
    def equalizer(counter):
        
        # Balance the dataset by removing over-represented samples from two arrays
        if total[star_rating] < maximum_amount:
            if len(a[index]) <= 50:
                if counter["50"] < no_of_words["50"]:
                    counter["50"] += 1
                    equalized_a.append(a[index])
                    equalized_b.append(star_rating)
                    total[star_rating] += 1
                    
            elif len(a[index]) > 50 and len(a[index]) <= 100:
                if counter["100"] < no_of_words["100"]:
                    counter["100"] += 1
                    equalized_a.append(a[index])
                    equalized_b.append(star_rating)
                    total[star_rating] += 1
                    
            elif len(a[index]) > 100 and len(a[index]) <= 150:
                if counter["150"] < no_of_words["150"]:
                    counter["150"] += 1
                    equalized_a.append(a[index])
                    equalized_b.append(star_rating)
                    total[star_rating] += 1
                    
            elif len(a[index]) > 150 and len(a[index]) <= 200:
                if counter["200"] < no_of_words["200"]:
                    counter["200"] += 1
                    equalized_a.append(a[index])
                    equalized_b.append(star_rating)
                    total[star_rating] += 1
                    
            elif len(a[index]) > 200 and len(a[index]) <= 250:
                if counter["250"] < no_of_words["250"]:
                    counter["250"] += 1
                    equalized_a.append(a[index])
                    equalized_b.append(star_rating)
                    total[star_rating] += 1
                    
            else:
                if counter["300"] < no_of_words["300"]:
                    counter["300"] += 1
                    equalized_a.append(a[index])
                    equalized_b.append(star_rating)
                    total[star_rating] += 1
                    
	# Balance data for each star rating
    for index, star_rating in enumerate(b):
        if star_rating == 1:
            equalizer(counter_1)
        elif star_rating == 2:
            equalizer(counter_2)
        elif star_rating == 3:
            equalizer(counter_3)
        elif star_rating == 4:
            equalizer(counter_4)
        else:
            equalizer(counter_5)

    return equalized_a, equalized_b

def actual_sentiment(data):
    
    # Convert star-rating into respective sentiments
    for i in data:
        if int(i[0]) == 4 or int(i[0]) == 5:
            return "positive"
        
        elif int(i[0]) == 1 or int(i[0]) == 2:
            return "negative"
        
        else:
            return "neutral"

def identify_token(text):
    
    # Return it's text back, as requested as a token
    return text

def train_and_evaluate(clf, X_train, X_test, y_train, y_test, accuracy_train, accuracy_test, precision_micro, recall_micro, f1_micro, precision_macro, recall_macro, f1_macro, precision_weight, recall_weight, f1_weight, ask_once_only):
    
    "--------- OVERFIT ------------------"
    # Overfit the training data using ADASYN
    if ask_once_only == False:
        ask_once_only = True
        false_ans_0 = True
        if ask_overfit == True:
            while false_ans_0 == True:
                user_input_0 = input("Do you want to overfit this dataset? (y/n) ")
                if user_input_0 == "y":
                    sm = ADASYN()
                    X_train, y_train = sm.fit_sample(X_train, y_train)
                    false_ans_0 = False
                    break
                else:
                    break
    "-----------------------------------------------------------"

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
	
    return accuracy_train, accuracy_test, precision_micro, recall_micro, f1_micro, precision_macro, recall_macro, f1_macro, precision_weight, recall_weight, f1_weight, ask_once_only

"--------------------------------------------------------------------------------------------------------------------"
"--------------------------------------------------------------------------------------------------------------------"

t1 = datetime.datetime.now()

# Read the data from Excel file & Data cleaning & pre-processing
filtered_dataset = preprocess.pre_process("tripadvisor_co_uk-travel_restaurant_reviews_sample.xlsx")

"--------- RUN THIS FOR - SENTIMENT ------------------"
run_sentiment = False
false_ans_1 = True
user_input_1 = input("Do you want to train & test using sentiments only? (y/n) ")
while false_ans_1 == True:
    if user_input_1 == "y":
        run_sentiment = True
        filtered_dataset["actual_sentiment"] = filtered_dataset.rating.apply(actual_sentiment)
        break
    elif user_input_1 == "n":
        break
"----------------------------------------------------"

print("Pre-process Time")
print(datetime.datetime.now() - t1)
print()

"--------------------------------------------------------------------------------------------------------------------"

# Storing both review texts and star ratings in repective arrays
review_text = []
star_rating = []
for index, row in filtered_dataset.iterrows():
    if run_sentiment == True:
        # Sentiment (Positive/Neutral/Negative)
        star_rating.append(row[8])
    else:
        # 1-5 stars
        star_rating.append(int(row[3][0]))
    tmp = []
    for word, tag in row[7]:
        tmp.append(word)
    review_text.append(tmp)

# Balance the data we are going to put into training and testing
ask_overfit = False
ask_once_only = False
false_ans_2 = True

"--------- BALANCE DATASET ------------------"
while false_ans_2 == True:
    user_input_2 = input("Do you want to balance dataset? (y/n) ")
    if user_input_2 == "y":      
        review_text, star_rating = class_equity(review_text, star_rating)
        break
    elif user_input_2 == "n":
        ask_overfit = True
        break
"--------- BALANCE DATASET ------------------"


# Vectorize review text into unigram, bigram and evaluates into a term document matrix of TF-IDF features
"--------- UNIGRAM / BIGRAM ------------------"
false_ans_3 = True
while false_ans_3 == True:
    user_input_3 = int(input("Choose either 1: Unigram || 2: Unigram & Bigram: "))
    if user_input_3 == 1 or user_input_3 == 2:
        false_ans_3 = False
        break
tfidf_vectorizer = TfidfVectorizer(tokenizer = identify_token, ngram_range = (1, user_input_3), lowercase = False)
"--------- UNIGRAM / BIGRAM ------------------"

t2 = datetime.datetime.now()

# Construct vocabulary and inverse document frequency from all the review texts
# Then, transform each review text into a tf-idf weighted document term matrix
vectorized_data = tfidf_vectorizer.fit_transform(review_text)

print("Vectorizing Time")
print(datetime.datetime.now() - t2)
print()

"--------------------------------------------------------------------------------------------------------------------" 
"--------- SVM / Random Forest ------------------"
false_ans_4 = True
while false_ans_4 == True:
    user_input_4 = int(input("Choose 1: Normal SVC || 2: Linear SVC || 3: RBF || 4: Random Forest "))
    if user_input_4 == 1 or user_input_4 == 2 or user_input_4 == 3 or user_input_4 == 4:
        false_ans_4 = False
        
if user_input_4 == 1:
    # Normal SVC
    svc = SVC(kernel = 'linear')
elif user_input_4 == 2:
    # LinearSVC
    svc = LinearSVC()
elif user_input_4 == 3:
    # RBF
    svc = SVC(kernel='rbf')
else:
    # Random Forest Classifier
    svc = RandomForestClassifier(n_estimators=1000)
"--------- SVM / Random Forest ------------------"
    
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
    accuracy_train, accuracy_test, precision_micro, recall_micro, f1_micro, precision_macro, recall_macro, f1_macro, precision_weight, recall_weight, f1_weight, ask_once_only = train_and_evaluate(svc, X_train, X_test, y_train, y_test, accuracy_train, accuracy_test, precision_micro, recall_micro, f1_micro, precision_macro, recall_macro, f1_macro, precision_weight, recall_weight, f1_weight, ask_once_only)
    
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

