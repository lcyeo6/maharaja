"""
@description:
    
    SVM-based Sentiment Analysis
@author: 
    
    Wan Chee Tin
    Yeo Lin Chung
    
"""

import datetime
import preprocess
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

def identify_token(text):
    
    # Return it's text back, as requested as a token
    return text

"--------------------------------------------------------------------------------------------------------------------"
"--------------------------------------------------------------------------------------------------------------------"

t1 = datetime.datetime.now()

# Read the data from Excel file & Data cleaning & pre-processing
filtered_dataset = preprocess.pre_process("tripadvisor_co_uk-travel_restaurant_reviews_sample.xlsx")

print("Pre-process Time")
print(datetime.datetime.now() - t1)
print()

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

# Vectorize review text into unigram and evaluates into a term document matrix of TF-IDF features
tfidf_vectorizer = TfidfVectorizer(tokenizer = identify_token, ngram_range = (1, 1), lowercase = False)

t2 = datetime.datetime.now()

# Construct vocabulary and inverse document frequency from all the review texts
# Then, transform each review text into a tf-idf weighted document term matrix
vectorized_data = tfidf_vectorizer.fit_transform(review_text)
pickle.dump(tfidf_vectorizer.vocabulary_, open("feature.pkl", "wb"))

print("Vectorizing Time")
print(datetime.datetime.now() - t2)
print()

"--------------------------------------------------------------------------------------------------------------------" 
# SVM (Kernel = Linear)
svc = SVC(kernel = 'linear')
   	
# Train and test the data
# Perform training on the training set
X_train = vectorized_data
y_train = star_rating
svc.fit(X_train, y_train)
    
# Save model into disk
filename = "finalized_model.sav"
pickle.dump(svc, open(filename, "wb"))


    