"""

@description:
    
    Lexicon-based Sentiment Analysis

@author: 
    
    Wan Chee Tin
    Yeo Lin Chung
    
"""

import preprocess
import lexicon
import lexicon_combined

# Read the data from Excel file & Data cleaning & pre-processing
filtered_dataset = preprocess.pre_process("tripadvisor_co_uk-travel_restaurant_reviews_sample.xlsx")

"-----LEXICON IMBALANCED DATASET XXX min per fold-----"
lexicon.run_lexicon(filtered_dataset)
"------------------------------------------------"

"-----LEXICON SENTIMENT IMBALANCED DATASET XXX min per fold-----"
lexicon_combined.run_lexicon(filtered_dataset)
"------------------------------------------------"