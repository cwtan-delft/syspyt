# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 16:26:38 2021

@author: tanch
"""

from assignment_B_model import logistic
from assignment_B_model import calibration

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% Import Data
excel = pd.read_excel("Goal13.xlsx","data")

#Data selection - 'Number of people affected by disaster (number)'
excel_affected = excel[excel["SeriesDescription"]=='Number of people affected by disaster (number)']


#%% Sanity Check
# sanity = pd.DataFrame(columns=["Country","Num. Values","Minimum","Maximum","Mean"])
sanity = []
#counting non-missing values per country
country_counts = excel_affected["GeoAreaName"].value_counts()
#print countries with the maximum amount of datapoints
# print([(k,v) for k,v in country_counts.items() if v == country_counts.max()])



for country in excel_affected["GeoAreaName"].unique():
    #no of values per country
    country_count = excel_affected.loc[excel_affected.GeoAreaName == country, 'GeoAreaName'].count()
#     #min
    country_min = excel_affected.loc[excel_affected.GeoAreaName == country, 'Value'].min()
#     #max
    country_max = excel_affected.loc[excel_affected.GeoAreaName == country, 'Value'].max()
#     #mean
    country_mean = np.mean(excel_affected.loc[excel_affected.GeoAreaName == country, 'Value'])
    
    sanity.append([country, country_count,country_min,country_max,country_mean])

sanity_df = pd.DataFrame(sanity, columns=["Country","Num. Values","Minimum","Maximum","Mean"])
sanity_df.set_index('Country')


#%% Logistic Model

logistic_results = {}
for country in excel_affected["GeoAreaName"].unique():
    logistic_results[country] = {}
    x = excel_affected.loc[excel_affected.GeoAreaName == country, 'TimePeriod']
    y = excel_affected.loc[excel_affected.GeoAreaName == country, 'Value']
    cf_result = calibration(x,y)
    lf_result = logistic(2030, cf_result[0], cf_result[1], cf_result[2], cf_result[3])
    logistic_results[country]['TimePeriod'] = x
    logistic_results[country]['Value'] = y
    logistic_results[country]['Calibration'] = cf_result
    logistic_results[country]['2030 Logistic'] = lf_result
    
    
