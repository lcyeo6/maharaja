"""

@description:
    
    Pre-process an excel file

@author: 
    
    Wan Chee Tin
    Yeo Lin Chung
    
"""

import re
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet

def wordnet_tag(tag):
    
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('R'):
        return wordnet.ADV
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('V'):
        return wordnet.VERB
    return None

def pre_process(sentence):    
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
        tokens = word_tokenize(letters)
        
        # Convert tokenized words into tokenized lower-case words
        small_letters = []
        for i in tokens:
            small_letters.append(i.lower())
        
        # Remove stop words 
        filtered = list(filter(lambda l: l not in stop_words, small_letters))
        
        # Parts of Speech Tagging
        tagged_filtered = pos_tag(filtered)
        
        # Lemmatize using WordNet's built-in function
        lemmatized_words = []
        for word, tag in tagged_filtered:
            wn_tag = wordnet_tag(tag)
            if wn_tag not in (wordnet.ADJ, wordnet.ADV, wordnet.NOUN):
                continue
            lemmatized_word = lemmatizer.lemmatize(word, pos = wn_tag)
            if not lemmatized_word:
                continue
            else:
                lemmatized_words.append((lemmatized_word, wn_tag))
            
        return lemmatized_words
    
    # Apply text_normalization function
    processed = text_normalization(sentence)
    
    return processed

#sentence = ["This is very bad food served by the restaurant. Their service is also sub par"]
#new = sentence[0]
#print(pre_process(new))