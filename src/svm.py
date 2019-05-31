"""
@description:
    
    Sentiment Analysis on SVM & Random Forest
    
@author: 
    
    Wan Chee Tin
    Yeo Lin Chung

@References:
    https://github.com/DeepmindHub/python-/blob/master/ROC%20Curve%20Multiclass.py
    https://medium.com/coinmonks/smote-and-adasyn-handling-imbalanced-data-set-34f5223e167
    https://medium.com/all-things-ai/in-depth-parameter-tuning-for-svc-758215394769
"""

import numpy as np
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

from sklearn.metrics import roc_curve, auc
import pandas as pd
from scipy import interp
from itertools import cycle
import matplotlib.pyplot as plt

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

def identify_token(text):
    
    # Return it's text back, as requested as a token
    return text

def train_and_evaluate(clf, X_train, X_test, y_train, y_test, accuracy_train, accuracy_test, precision_weight, recall_weight, f1_weight, ask_once_only):
    
    # Overfit the training data using ADASYN
    if ask_once_only == False:
        ask_once_only = True
        false_ans_0 = True
        if ask_overfit == True:
            while false_ans_0 == True:
                user_input_0 = input("Do you want to overfit this dataset? (y/n)\n")
                if user_input_0 == "y":
                    sm = ADASYN()
                    X_train, y_train = sm.fit_sample(X_train, y_train)
                    false_ans_0 = False
                else:
                    break

    # Perform training on the training set
    clf.fit(X_train, y_train)

    accuracy_train.append(clf.score(X_train, y_train))
    accuracy_test.append(clf.score(X_test, y_test))
    
	# Predicting the training data used on the test data
    y_pred = clf.predict(X_test)

    "--------------------------------------------------------------------------------------------------------------------"
# ROC Curve plotting
    n_classes = 5
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(np.array(pd.get_dummies(y_test))[:, i], np.array(pd.get_dummies(y_pred))[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    lw=2
    plt.figure(figsize=(10,7))
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='green', linestyle=':', linewidth=4)
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'pink', 'green'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--',color='red', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.annotate('Random Guess',(.5,.42),color='red')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic for RandomForestClassifier(n_estimators = 250)')
    plt.legend(loc="lower right")
    plt.show()

    "--------------------------------------------------------------------------------------------------------------------"

# Classification Report
    print ("Classification Report:")
    print (classification_report(y_test, y_pred))
# Confusion Matrix
    print ("Confusion Matrix:")
    print (confusion_matrix(y_test, y_pred))
    print()

# Append all results into lists
    precision_weight.append(precision_score(y_test,y_pred,average = 'weighted', labels = np.unique(y_pred)))
    recall_weight.append(recall_score(y_test, y_pred, average = 'weighted'))
    f1_weight.append(f1_score(y_test, y_pred, average = 'weighted', labels = np.unique(y_pred)))
	
    return accuracy_train, accuracy_test, precision_weight, recall_weight, f1_weight, ask_once_only

"-------------------------------------------------------------------------------------------------------------------"
"-------------------------------------------------------------------------------------------------------------------"


"---------------------------------------------------- PRE-PROCESS ----------------------------------------------------"
t1 = datetime.datetime.now()

# Read the data from Excel file & Data cleaning & pre-processing
filtered_dataset = preprocess.pre_process("tripadvisor_co_uk-travel_restaurant_reviews_sample.xlsx")

print("\nPre-process Time")
print(datetime.datetime.now() - t1) 

"--------------------------------------------------------------------------------------------------------------------"


# Storing both review texts and star ratings in repective arrays
review_text = []
star_rating = []
for index, row in filtered_dataset.iterrows():
    star_rating.append(int(row[3][0]))
    tmp = []
    for word, tag in row[7]:
        tmp.append(word)
    review_text.append(tmp)



"------------------------------------------------- BALANCE DATASET -------------------------------------------------"
# Balance the data we are going to put into training and testing
ask_overfit = False
ask_once_only = False

while True:
    input_balance = input("Do you want to balance dataset? (y/n)\n")
    if input_balance == "y":      
        review_text, star_rating = class_equity(review_text, star_rating)
        break
    elif input_balance == "n":
        ask_overfit = True
        break
    
"-------------------------------------------------------------------------------------------------------------------" 



"------------------------------------------------------ TFIDF ------------------------------------------------------"
# Vectorize review text into unigram, bigram and evaluates into a term document matrix of TF-IDF features
while True:
    input_ngram = int(input("Choose either 1: Unigram || 2: Bigram:\n"))
    if input_ngram == 1 or input_ngram == 2:
        break
        
tfidf_vectorizer = TfidfVectorizer(tokenizer = identify_token, ngram_range = (input_ngram, input_ngram), lowercase = False)

t2 = datetime.datetime.now()

# Construct vocabulary and inverse document frequency from all the review texts
# Then, transform each review text into a tf-idf weighted document term matrix
vectorized_data = tfidf_vectorizer.fit_transform(review_text)

print("\nVectorizing Time")
print(datetime.datetime.now() - t2)

"-------------------------------------------------------------------------------------------------------------------" 



"----------------------------------------------- SVM / RANDOM FOREST -----------------------------------------------"
while True:
    input_model = int(input("Choose 1: Normal SVC || 2: Linear SVC || 3: Random Forest\n"))
    if input_model == 1 or input_model == 2 or input_model == 3:
        break
        
# SVC(kernel = 'linear')
if input_model == 1:
    model = SVC(kernel = 'linear')
    
# LinearSVC
elif input_model == 2:             
    model = LinearSVC()
    
# Random Forest Classifier
else:
    model = RandomForestClassifier(n_estimators = 250)
    
"-------------------------------------------------------------------------------------------------------------------"

accuracy_train = []
accuracy_test = []

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
    accuracy_train, accuracy_test, precision_weight, recall_weight, f1_weight, ask_once_only = train_and_evaluate(model, X_train, X_test, y_train, y_test, accuracy_train, accuracy_test, precision_weight, recall_weight, f1_weight, ask_once_only)
    
    print("Train and Test Time")
    print(datetime.datetime.now() - t3)
    print()

# Print out the average scores 
print("accuracy train:   {}".format(np.mean(accuracy_train)))
print("accuracy test:    {}".format(np.mean(accuracy_test)))
print()

print("precision weight: {}".format(np.mean(precision_weight)))
print("recall weight:    {}".format(np.mean(recall_weight)))
print("f1 weight:        {}".format(np.mean(f1_weight)))