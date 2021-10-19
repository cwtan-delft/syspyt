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
from scipy.stats import pearsonr
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


unique_countries = excel_affected["GeoAreaName"].unique()
for country in unique_countries:
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

projection_year = 2030
logistic_results = {}

for country in excel_affected["GeoAreaName"].unique():
    logistic_results[country] = {}
    x = excel_affected.loc[excel_affected.GeoAreaName == country, 'TimePeriod']
    y = excel_affected.loc[excel_affected.GeoAreaName == country, 'Value']
    start, K, x_peak, r = calibration(x,y)

    lf_2030 = logistic(projection_year, start, K, x_peak, r)
    logistic_results[country]['TimePeriod'] = x
    logistic_results[country]['Value'] = y
    logistic_results[country]['Calibration'] = (start, K, x_peak, r)
    logistic_results[country]['2030 Logistic'] = lf_2030
    logistic_results[country]['Growth Rate'] = r
    
    #generating logistic model results for datapoints
    lf_list = [logistic(year, start, K, x_peak, r) for year in x]
    logistic_results[country]['Logistic Series'] = lf_list

    
    
#performing Linear Regression
print("\n*Regression")
# calculating r2 with McFadden's R2
r_squared_McFadden = 1 - (np.log()/np.log())
# calculating r2 with Cox and Snell's R2
# print('\n*PBIAS')
# lr_pbias = percentBias(reg_year_temp[:,1],lr_array[:,1])
# print('\n*NRMSE')
# lr_nrmse = math.sqrt(mean_squared_error(reg_year_temp[:,1],lr_array[:,1]))/np.std(reg_year_temp[:,1])
# lr_mse,lr_rmse,lr_nrmse = nrmse(reg_year_temp[:,1],lr_array[:,1])

