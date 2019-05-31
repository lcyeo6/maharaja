"""
@description:
    
    SVM Saving our Predicting Model
@author: 
    
    Wan Chee Tin
    Yeo Lin Chung

@References:
    https://stackoverflow.com/questions/29788047/keep-tfidf-result-for-predicting-new-content-using-scikit-for-python
    https://machinelearningmastery.com/make-predictions-scikit-learn/
    https://machinelearningmastery.com/train-final-machine-learning-model/
    https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
    
"""

import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import preprocess_user_input

def identify_token(text):
    
    # Return it's text back, as requested as a token
    return text

def predict(user_input):
    
    review_text = []
    # Get the user input from user interface
    processed = preprocess_user_input.pre_process(user_input)
    tmp = []
    for word, tag in processed:
        tmp.append(word)
    review_text.append(tmp)
    
    filename = "finalized_model.sav"
    
    # Load the saved features
    loaded_tfidf = TfidfVectorizer(tokenizer = identify_token, ngram_range = (1, 1), lowercase = False, vocabulary = pickle.load(open("feature.pkl", "rb")))
    X_new = loaded_tfidf.fit_transform(review_text)
    # Load the saved model
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.predict(X_new)

    return result