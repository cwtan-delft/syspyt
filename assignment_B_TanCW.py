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
from scipy.stats import linregress
from sklearn.model_selection import KFold

import traceback

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
    '''
    

    Parameters
    ----------
    obs_array : TYPE
        DESCRIPTION.
    sim_array : TYPE
        DESCRIPTION.

    Returns
    -------
    r2_value : TYPE
        DESCRIPTION.
    pbias_value : TYPE
        DESCRIPTION.
    nrmse_value : TYPE
        DESCRIPTION.

    '''
    if len(obs_array) < 2:
        r2_value = pbias_value = nrmse_value = math.nan
        
    else: 
        r_value, p_value = pearsonr(obs_array, sim_array)
        r2_value = r_value**2
        
        pbias_value = percent_bias(obs_array, sim_array)
        
        mse_value, rmse_value, nrmse_value = nrmse(obs_array, sim_array)
    
    return r2_value, pbias_value, nrmse_value

def fold_logistic(timeperiod, value, folds=5, random_seed=1):
    '''
    

    Parameters
    ----------
    timeperiod : TYPE
        DESCRIPTION.
    value : TYPE
        DESCRIPTION.
    folds : TYPE, optional
        DESCRIPTION. The default is 5.
    random_seed : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    None.

    '''
    
    length_t = len(timeperiod)
    length_v = len(timeperiod)
    length_min = min(length_v, length_t)
    if length_t != length_v:
        print("Length of inputs are not equal.")
    length_t = len(timeperiod)
    length_v = len(timeperiod)
    length_min = min(length_v, length_t)
    
    if length_t != length_v:
        print("Length of inputs are not equal.")
        return np.empty(3)*np.nan
    elif length_min < folds:
        print("->{} dataset ({}) smaller than number of folds({})!".format(country,length_min,folds))
        return np.empty(3)*np.nan
    else:
        fold_dataset = pd.concat([timeperiod,value], axis =1)
        fold_dataset.reset_index(drop=True)
        # print(fold_dataset)
        kfold = KFold(folds,shuffle=True,random_state = random_seed)
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
            
            return np.array([r2_averaged, pbias_averaged, nrmse_averaged])
        except RuntimeError:
            print("->No solution for fold logistic found for {}".format(country))
            return np.empty(3)*np.nan

def plot_obs_sim (country, results_dict):
    #initialise plot
    fig = plt.figure(figsize = (6,4))
    plt.xlabel("Year", fontsize=10)  
    plt.ylabel("Number of people affected by disaster", fontsize=10)
    
    # list of tableau colours 
    tableau10 = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    
    #create variables for plotting
    timeperiod = results_dict[country]['TimePeriod']
    obs = results_dict[country]['Value']
    log_sim = results_dict[country]['Logistic Series']
    lin_sim = results_dict[country]['Linear Series']
    
    #plot functions
    plt.scatter(timeperiod, obs, color = tableau10[0], label="{}: Observed".format(country),  )
    plt.plot(timeperiod, log_sim, color = tableau10[1], label="{}: Logistic Simulated".format(country), linestyle = "dashed" )
    plt.plot(timeperiod, lin_sim, color = tableau10[2], label="{}: Linear Simulated".format(country) )
    plt.legend(loc='best', fontsize= 10)
    # plt.xlim(min(timeperiod),max(timeperiod))
    plt.xticks(np.arange(2004,2021,2))
    plt.ylim(min(min(obs),min(lin_sim),min(log_sim)))
    
    plt.tight_layout()
    plt.show()
       
    return fig 

def plot_growthrate (df, country_column, growth_column, proj_column):
    fig = plt.figure(figsize = (10,8))
    plt.scatter(df[growth_column], df[country_column], c=df[proj_column], cmap = 'brg' ,)
    plt.colorbar(pad = 0.02, label="Projected Number of people affected by disaster in 2030")
    
    plt.xlabel("Growth Rate", fontsize = 10)
    plt.xlim(-10.5,10.5)
    plt.tight_layout()
    plt.show()
    return fig
#%% Import Data

# SDG 13.1.1
goal13 = pd.read_excel("Goal13.xlsx","data")
metric = 'Number of people affected by disaster (number)'

#Data selection - 'Number of people affected by disaster (number)'
goal13_affected = goal13[goal13["SeriesDescription"]==metric]

# create pivot table of country vs year for number of people affected by disaster
pivot13 = goal13_affected.pivot(index=list(goal13_affected)[0:7], columns="TimePeriod", values="Value")

# Population 
world_pop = pd.read_excel("WPP2019_POP_F01_1_TOTAL_POPULATION_BOTH_SEXES.xlsx","ESTIMATES", header = 16, index_col=0)
country_pop = world_pop.loc[world_pop.Type == "Country/Area"]
#top 30 largest countries
country_pop_t30 = country_pop.sort_values("2020", ascending= False).head(30)


    
#%% Sanity Check (Old)

sanity = []
#counting non-missing values per country
country_counts = goal13_affected["GeoAreaCode"].value_counts()
#print countries with the maximum amount of datapoints
# print([(k,v) for k,v in country_counts.items() if v == country_counts.max()])


unique_countries = goal13_affected["GeoAreaCode"].unique()
for country in unique_countries:
    country_name = goal13_affected.loc[goal13_affected.GeoAreaCode == country, 'GeoAreaName'].unique()[0]
    #no of values per country
    country_count = goal13_affected.loc[goal13_affected.GeoAreaCode == country, 'GeoAreaName'].count()
#     #min
    country_min = goal13_affected.loc[goal13_affected.GeoAreaCode == country, 'Value'].min()
#     #max
    country_max = goal13_affected.loc[goal13_affected.GeoAreaCode == country, 'Value'].max()
#     #mean
    country_mean = np.mean(goal13_affected.loc[goal13_affected.GeoAreaCode == country, 'Value'])
    
    sanity.append([country, country_name, country_count,country_min,country_max,country_mean])

sanity_df = pd.DataFrame(sanity, columns=["GeoAreaCode","GeoAreaName","Observations","Minimum","Maximum","Mean"])
sanity_df.set_index( 'GeoAreaCode')

sanity_by_max = sanity_df[sanity_df.Observations >=10].sort_values("Maximum", ascending = False)
#%% Logistic  (old)

#set variables for logistic evaluation and validation here
projection_year = 2030
num_folds = 5
k_fold_seed = 1
results = {}
results_30 = {}


# for country in goal13_affected["GeoAreaName"].unique()[5:10]:
for idx, country in enumerate(goal13_affected["GeoAreaName"].unique()):
    #for debug
    #print(country)
    
    #initialise values and dictionaries
    results[country] = {}
    x = goal13_affected.loc[goal13_affected.GeoAreaName == country, 'TimePeriod']
    y = goal13_affected.loc[goal13_affected.GeoAreaName == country, 'Value']
    

    try:
        #run logistic calibration
        start, K, x_peak, r = calibration(x,y)
        log_2030 = logistic(projection_year, start, K, x_peak, r)
        
        #generating logistic model results for datapoints
        log_list = [logistic(year, start, K, x_peak, r) for year in x]
        
        #performing Evaluation of regression
        log_r2, log_pbias, log_nrmse = evaluation(y,log_list)
        
        if math.isnan(log_pbias):
            print("->{} has {} datapoints, not enough for regression".format(country,len(log_list)))
            
    except RuntimeError:
        print("->No solution for logistic found for {}".format(country))
        start = K = x_peak = r = log_2030 = log_list = log_pbias = log_pbias = log_nrmse = math.nan 
        
        pass
    
    
    try:
        #run linear regression
        slope, intercept, r_value, p_value, std_err = linregress(x,y)
        #generating linear model results
        lin_list = [slope*year + intercept for year in x]
        lin_2030 = slope*projection_year +intercept
        #performing evaluation of regression
        lin_r2, lin_pbias, lin_nrmse = evaluation(y,lin_list)
        
    except Exception:
        print(traceback.format_exc())
        print("\nLinear regression issue")
    
    
    #append values to results dictionary
    results[country]['TimePeriod'] = x.tolist()
    results[country]['Value'] = y.tolist()
    results[country]['Calibration'] = (start, K, x_peak, r)
    
    results[country]['Logistic Series'] = log_list
    results[country]['Linear Series'] = lin_list
    
    results[country]['Growth Rate'] = r
    results[country]['2030 Logistic'] = log_2030
    results[country]['2030 Linear'] = lin_2030
    
    results[country]['Logistic evaluation R2'] = log_pbias
    results[country]['Logistic evaluation PBIAS'] = log_pbias
    results[country]['Logistic evaluation NRMSE'] = log_nrmse
    
    results[country]['Linear evaluation R2'] = lin_pbias
    results[country]['Linear evaluation PBIAS'] = lin_pbias
    results[country]['Linear evaluation NRMSE'] = lin_nrmse
    
    fold_dataset = pd.concat([x,y], axis =1)
    fold_dataset.reset_index(drop=True)
    fold_results = fold_logistic(x,y,num_folds, k_fold_seed)
    results[country]['{}-Fold validation mean R2'], results[country]['{}-Fold validation mean R2'.format(num_folds)], results[country]['{}-Fold validation mean R2'.format(num_folds)] = fold_results
    
    
    # #append values to plotting dictionary
    # if country in country_pop_t30["Region, subregion, country or area *"].tolist():
    #     results_30[country] = {}
    #     results_30[country]['2030 Logistic'] = log_2030       
    #     results_30[country]['2030 Linear'] = lin_2030
    #     results_30[country]['Growth Rate'] = r       
    
#%% Plotting

#list of countries for plotting
country_plot =  ["Niger", "Sri Lanka", "Peru", ]

#plot
for country in country_plot:
    fig = plot_obs_sim(country,results)
    plt.close(fig)
    
#%% 30 most polulous countries
# "['Nigeria', 'Democratic Republic of the Congo', 'Germany', 'United Kingdom', 'Italy', 'Spain'] not in index"
#preparation of dataset

results_df = pd.DataFrame.from_dict(results,orient='index')
results_df.index.name = 'GeoAreaName'

results_30_df = pd.merge(results_df,country_pop_t30[["Region, subregion, country or area *","2020"]], left_index=True, right_on = "Region, subregion, country or area *", how="right")
results_30_df.sort_values("2020", ascending = False, inplace=True)
results_30_df.rename(columns = {"Region, subregion, country or area *": 'GeoAreaName', "2020":"2020 Population"}, inplace=True)
results_30_df.set_index('GeoAreaName')
plot_df = results_30_df[["GeoAreaName","2020 Population", "2030 Logistic", "2030 Linear", "Growth Rate"]]

#plot
growthplot = plot_growthrate(plot_df, "GeoAreaName", "Growth Rate", "2030 Logistic")
# growthplot.savefig("ordered dot plot.png")
plt.close(growthplot)
#%% create csv file
csv_name = "goal 13 analysis main results.csv"

results_df = pd.DataFrame.from_dict(results,orient='index')
results_df.index.name = 'GeoAreaName'
csv_indexes = pd.DataFrame(index=pivot13.index)
csv_data = pd.merge(csv_indexes, results_df[list(results_df)[-11:-1]], left_index=True, right_index=True, how='left')
# csv_data.to_csv(csv_name)