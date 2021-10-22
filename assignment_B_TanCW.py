# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 16:26:38 2021

@author: tanch
"""

from assignment_B_model import logistic
from assignment_B_model import calibration

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.model_selection import KFold

#%% functions
# function to calculated percentage bias
def percent_bias(obs_array,sim_array):
    '''
    Arguments:np array of observed values, np array of simulated values \n
    Returns: PBIAS value of simulated values
    '''
    if len(obs_array) != len(sim_array):
        print("percentBias: !!! inputs do not have the same length")
        return
    else:    
        pbias = 100*(sum(sim_array-obs_array)/sum(obs_array))
    return pbias

# function to calculate variations of mean square error (mse): mse, root mse (rmse), normalised rmse (nrmse)
def nrmse(obs_array,sim_array):
    '''
    Arguments:np array of observed values, np array of simulated values \n
    Returns: mean-square-error,root-mean-square-error,normalised-root-mean-square-error(std dev) \n
    '''
    if len(obs_array) != len(sim_array):
        print("nrmse: !!! inputs do not have the same length")
        return
    else:

        mse_val = np.sum((sim_array - obs_array)**2)/len(obs_array)
        rmse_val = math.sqrt(mse_val)
        nrmse_val = rmse_val/np.std(obs_array)
    
    return mse_val,rmse_val,nrmse_val

def evaluation (obs_array, sim_array):
    if len(obs_array) < 2:
        r2_value = pbias_value = nrmse_value = math.nan
        
    else: 
        r_value, p_value = pearsonr(obs_array, sim_array)
        r2_value = r_value**2
        
        pbias_value = percent_bias(obs_array, sim_array)
        
        mse_value, rmse_value, nrmse_value = nrmse(obs_array, sim_array)
    
    return r2_value, pbias_value, nrmse_value

def fold_logistic(time, value, folds=5):
    if len(time) < folds or len(value) < folds:
        print("->Dataset smaller than number of folds!")
        return
    else:
        fold_dataset = pd.concat([time,value], axis =1)
        fold_dataset.reset_index(drop=True)
        # print(fold_dataset)
        kfold = KFold(folds)
        splits = [i for i in kfold.split(fold_dataset)]
        # print(splits)
        
        
        fold_data = []

        try: 
            for foldset in splits:
                train, test = foldset
                # print("Train:", train)
                # print("Test:", test)
                
                # print(fold_dataset.iloc[4])
                train_set = fold_dataset.iloc[train]
                # print(train_set)
        
                start, K, x_peak, r = calibration(train_set['TimePeriod'],train_set['Value'])
                
                test_set = fold_dataset.iloc[test]
                # print(test_set['TimePeriod'])
                # print("Test set:", test_time)
                lf_results = [logistic(year, start, K, x_peak, r) for year in test_set['TimePeriod']]
                r2_value, pbias_value, nrmse_value = evaluation(test_set["Value"],lf_results)
                fold_data.append([r2_value, pbias_value, nrmse_value])
                
            # print(fold_data)
            r2_averaged= np.mean(fold_data[:][0])
            pbias_averaged= np.mean(fold_data[:][1])
            nrmse_averaged= np.mean(fold_data[:][2])
            
            return (r2_averaged, pbias_averaged, nrmse_averaged)
        except RuntimeError:
            print("->No solution for logistic found for {}".format(country))
            return

            
#%% Import Data
excel = pd.read_excel("Goal13.xlsx","data")

metric = 'Number of people affected by disaster (number)'

#Data selection - 'Number of people affected by disaster (number)'
excel_affected = excel[excel["SeriesDescription"]==metric]
# create pivot table of country vs year for number of people affected by disaster
pivot = excel_affected.pivot(index="GeoAreaName", columns="TimePeriod", values="Value")


#%% Sanity Check (Old)
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


#%% Logistic  (old)

projection_year = 2030
logistic_results = {}

# for country in excel_affected["GeoAreaName"].unique()[5:10]:
for country in excel_affected["GeoAreaName"].unique():
    #for debug
    print(country)
    
    #initialise values and dictionaries
    logistic_results[country] = {}
    x = excel_affected.loc[excel_affected.GeoAreaName == country, 'TimePeriod']
    y = excel_affected.loc[excel_affected.GeoAreaName == country, 'Value']
    

    try:
        #run logistic calibration
        start, K, x_peak, r = calibration(x,y)
        lf_2030 = logistic(projection_year, start, K, x_peak, r)
        
        #generating logistic model results for datapoints
        lf_list = [logistic(year, start, K, x_peak, r) for year in x]
        
        #performing Evaluation of regression
        r2_value, pbias_value, nrmse_value = evaluation(y,lf_list)
        
        if math.isnan(r2_value):
            print("->{} has {} datapoints, not enough for regression".format(country,len(lf_list)))
            start = K = x_peak = r = lf_2030 = lf_list = r2_value = pbias_value = nrmse_value = math.nan 
    except RuntimeError:
        print("->No solution for logistic found for {}".format(country))
        
        pass
    #append values to dictionary
    logistic_results[country]['TimePeriod'] = x.tolist()
    logistic_results[country]['Value'] = y.tolist()
    logistic_results[country]['Calibration'] = (start, K, x_peak, r)
    logistic_results[country]['2030 Logistic'] = lf_2030
    logistic_results[country]['Growth Rate'] = r
    
    logistic_results[country]['Logistic Series'] = lf_list
    
    logistic_results[country]['R2'] = r2_value
    logistic_results[country]['PBIAS'] = pbias_value
    logistic_results[country]['Calibration'] = nrmse_value
    
    fold_dataset = pd.concat([x,y], axis =1)
    fold_dataset.reset_index(drop=True)
    fold_results = fold_logistic(x,y,5)
    logistic_results[country]['Fold results'] = fold_results
    
#%%

#list of countries for plotting
country_plot =  ["Australia", "Mexico"]
# list of tableau colours 
tableau10 = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

#initialise plot
fig = plt.figure()
ax = plt.subplot(111)


for idx,country in enumerate(country_plot):
    time = logistic_results[country]['TimePeriod']
    obs = logistic_results[country]['Value']
    sim = logistic_results[country]['Logistic Series']
    
    ax.plot(time,obs, color = tableau10[idx*2], label="{}: Observed".format(country) )
    ax.plot(time,sim, color = tableau10[idx*2+1], label="{}: Simulated".format(country) )
    
    
plt.legend()    
plt.xlabel("Year", fontsize=10)  
plt.ylabel(metric, fontsize=12)   