"""

@description:
    
    SVM-based Sentiment Analysis

@author: 
    
    Wan Chee Tin
    Yeo Lin Chung
    
"""

import pandas as pd
import re
from collections import Counter

def read_excel(filename):
    
    dataset = pd.read_excel(filename, sheet_name = "Sheet1")
    
    return dataset

def pre_process(dataset):
    
    # Remove unwanted columns
    dataset = dataset.drop(["uniq_id", "url", "restaurant_id", "restaurant_location", "category", "review_date", "author", "author_url", "location", "visited_on"], axis = 1)
    
    # Remove rows if NaN exists in specific column, in this case "review_text" & "rating" columns
    dataset = dataset.dropna(subset = ["review_text", "rating"])
    
    def text_normalization(data):
        
        # Replace unknown characters and numbers with a space
        # [^a-zA-Z] will match any character except lower-case and upper-case letters
        data = re.sub("[^a-zA-Z]", " ", data)
        
        return data
        
    # Apply text_normalization function
    dataset["normalized_review_text"] = dataset.review_text.apply(text_normalization)
    
    return dataset

def class_equity(a, b):
    
    # Count the frequency of each category
    occurrence = Counter(b)
    
    # The least common category will be the maximum equity
    maximum = occurrence.most_common()[-1][1]
    
    # Serve as counter
    total = dict()
    for category in occurrence.keys():
        total[category] = 0
    
    equalized_a = []
    equalized_b = []
    
    # Balance the dataset by removing over-represented samples from two arrays
    for index, element in enumerate(b):
        if total[element] < maximum:
            equalized_a.append(a[index])
            equalized_b.append(element)
            total[element] += 1
    
    return equalized_a, equalized_b

# Read the data from Excel file
dataset = read_excel("tripadvisor_co_uk-travel_restaurant_reviews_sample.xlsx")

# Data cleaning & pre-processing
filtered_dataset = pre_process(dataset)

# Storing both normalized review texts and star ratings in repective arrays
review_text = []
star_rating = []
for index, row in filtered_dataset.iterrows():
    review_text.append(row[7])
    star_rating.append(int(row[3][0]))
    
equalized_review_text, equalized_star_rating = class_equity(review_text, star_rating)
    