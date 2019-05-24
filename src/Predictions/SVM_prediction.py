"""
@description:
    
    SVM-based Sentiment Analysis
@author: 
    
    Wan Chee Tin
    Yeo Lin Chung
    
"""

import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import preprocess_user_input

def identify_token(text):
    
    # Return it's text back, as requested as a token
    return text

review_text = []
user_input = input("Enter your review: ")
processed = preprocess_user_input.pre_process(user_input)
tmp = []
for word, tag in processed:
    tmp.append(word)
review_text.append(tmp)

filename = "finalized_model.sav"

loaded_tfidf = TfidfVectorizer(tokenizer = identify_token, ngram_range = (1, 1), lowercase = False, vocabulary = pickle.load(open("feature.pkl", "rb")))
X_new = loaded_tfidf.fit_transform(review_text)
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.predict(X_new)
print(result)