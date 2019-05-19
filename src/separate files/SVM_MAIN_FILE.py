"""
@description:
    
    SVM-based Sentiment Analysis
@author: 
    
    Wan Chee Tin
    Yeo Lin Chung
    
"""

import datetime
import preprocess
import svm
import svm_balanced
import svm_overfit
import svm_combined
import svm_balanced_combined
import svm_overfit_combined

"--------------------------------------------------------------------------------------------------------------------"

t1 = datetime.datetime.now()

# Read the data from Excel file & Data cleaning & pre-processing
filtered_dataset = preprocess.pre_process("tripadvisor_co_uk-travel_restaurant_reviews_sample.xlsx")

"-----SVM IMBALANCED DATASET 6-7 min per fold-----"
#svm.run_svm(filtered_dataset)
"------------------------------------------------"

"-----SVM BALANCED DATASET 2-3 min per fold-----"
#svm_balanced.run_svm(filtered_dataset)
"------------------------------------------------"

"-----SVM OVERFIT DATASET XXX min per fold-----"
#svm_overfit.run_svm(filtered_dataset)
"------------------------------------------------"

"-----SVM IMBALANCED SENTIMENT DATASET XXX min per fold-----"
#svm_combined.run_svm(filtered_dataset)
"-----------------------------------------------------------"

"-----SVM BALANCED SENTIMENT DATASET XXX min per fold-----"
#svm_balanced_combined.run_svm(filtered_dataset)
"-----------------------------------------------------------"

"-----SVM OVERFIT SENTIMENT DATASET XXX min per fold-----"
svm_overfit_combined.run_svm(filtered_dataset)
"-----------------------------------------------------------"