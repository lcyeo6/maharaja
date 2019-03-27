# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 23:08:22 2019

@author:
        
    Wan Chee Tin
    Yeo Lin Chung
    
"""
import pandas as pd

file = "tripadvisor_co_uk-travel_restaurant_reviews_sample.xlsx"

dataset = pd.read_excel(file, sheet_name = "Sheet_1")

    
print()