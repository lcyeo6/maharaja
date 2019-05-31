# MAHARAJA 
Project is run in Anaconda - Spyder

## Review text Prediction - User Interface
### Files needed
frontend.py, preprocess_user_input, svm_prediction.py, lexicon_prediction.py

### How to run
1. Open and run frontend.py
2. A new window will appear. 
3. Input review text that you want to predict.
4. Click on Lexicon to predict sentiment
5. Click on SVM to predict star rating
6. Click on Clear to clear input box and results


## Train and test 
### Rule based model classifier - Lexicon MPQA & SentiWordNet
#### Files needed
preprocess.py & lexicon.py

#### How to run
Open and run lexicon.py file, in this file, you will need to choose 

MPQA or SentiWordNet **Answer only 1 or 2**

Once chosen, let the program run and results will be shown in the console.


### Machine Learning classifier - SVM & Random Forest
#### Files needed
preprocess.py & svm.py

#### How to run
Open and run svm.py, in this file, you need to choose
1. Do you want to balance dataset? **Answer only "y" or "n"**

2. Choose either run in: 1. Unigram or 2. Bigram **Answer only 1 or 2**

3. Choose which classifier: 1. SVC(kernel='linear') or 2. LinearSVC() or 3. Random Forest **Answer only 1 or 2 or 3**

4. IF YOU ANSWER "n" in Question 1, Do you want to oversample the dataset? **Answer only "y" or "n"**

Once all are chosen, let the program run and results will be shown in the console.

## Unit test files
unit_test_pre_prrocess & unit_test_lexicon