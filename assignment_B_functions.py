# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 03:35:54 2021

@author: tanch
"""
from assignment_B_model import logistic
from assignment_B_model import calibration

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

# import traceback

#%% functions

# function to calculated percentage bias
def percent_bias(obs_array,sim_array):    
    
    '''
    Pecentage bias
    
    Takes an array of observed values and simulated values and calculated and returns the percent bias (PBIAS) value of the simulated values. \n
    
    Parameters
    ----------
    obs_array : array-like
        Array of observed values.
    sim_array : array-like
        Array of simulated values from modelling.

    Returns
    -------
    pbias : float
        PBIAS value of simulated array.
    
    Warning
    -------
    Returns None if the length of the parameters do not match.
    '''

    if len(obs_array) != len(sim_array):
        print("percentBias: !!! inputs do not have the same length")
        return None
    else:    
        pbias = 100*(sum(sim_array-obs_array)/sum(obs_array))
    return pbias

# function to calculate variations of mean square error (mse): mse, root mse (rmse), normalised rmse (nrmse)
def nrmse(obs_array,sim_array):
    '''
    Normalised root mean square error
    
    Takes an array of observed values and simulated values and
    calculates and returns the mean-square-error (MSE), root-mean-square-error (RMSE),
    normalised-root-mean-square-error(NRMSE, from std dev) values of the simulated values.

    Parameters
    ----------
    obs_array : array-like
        Array of observed values.
    sim_array : array-like
        Array of simulated values from modelling.

    Returns
    -------
    mse_val : float
        MSE value of simulated array.
    rmse_val : float
        RMSE value of simulated array.
    nrmse_val : float
        NRMSE value of simulated array.

    Warning
    -------
    Returns None if the length of the parameters do not match.
    '''
    
    if len(obs_array) != len(sim_array):
        print("nrmse: !!! inputs do not have the same length")
        return
    else:

        mse_val = np.sum((sim_array - obs_array)**2)/len(obs_array)
        rmse_val = math.sqrt(mse_val)
        nrmse_val = rmse_val/np.std(obs_array)
    
    return mse_val,rmse_val,nrmse_val


def r2(obs_array,sim_array):
    '''
    R-squared value
    
    Takes an array of observed values and simulated values and
    calculates and returns the R-squared value for the simulated values.   
    
    Parameters
    ----------
    obs_array : Array-like
        Array of observed values.
    sim_array : Array-like
        Array of simulated values from modelling.


    Returns
    -------
    r2_val : float
        R-squared value for the simulated array.
        
    Warning
    -------
    Returns None if the length of the arrays do not match.

    '''
    
    if len(obs_array) != len(sim_array):
        print("percentBias: !!! inputs do not have the same length")
        return
    else:    
        yhat = sim_array
        ybar = np.sum(obs_array)/len(obs_array)
        ssreg = np.sum((yhat-ybar)**2)
        sstot = np.sum((obs_array - ybar)**2)
        r2_val = ssreg / sstot
        return r2_val
    

def evaluation (obs_array, sim_array):
    '''
    3-criteria evaluation
    
    Takes an array of observed values and simulated values,
    calculates and returns the criteria evaluation of R^2, Percentage-bias and
    Normalised root mean square error values.

    Parameters
    ----------
    obs_array : array-like
        Array of observed values.
    sim_array : array-like
        Array of simulated values from modelling.

    Returns
    -------
    r2_value : float
        R^2 score (coefficient of determination) regression score of simulated values based on the the scikit learn metrics method.
    pbias_value : float
        Calculated Pecentage-bias of simulated values.
    nrmse_value : float
        Calculated Normalised root mean square error value of simulated values.
    
    Warning
    -------
    Returns NaN for all variables if the length of the arrays do not match.
    '''
    if len(obs_array) < 2:
        r2_value = pbias_value = nrmse_value = math.nan
        
    else:
        r2_value = r2_score(obs_array, sim_array)
        
        # r2_value = r2(obs_array, sim_array)
        
        pbias_value = percent_bias(obs_array, sim_array)
        
        mse_value, rmse_value, nrmse_value = nrmse(obs_array, sim_array)
    
    return r2_value, pbias_value, nrmse_value

def fold_logistic(timeperiod, value, country, folds=5, random_seed=1):
    '''
    K-fold validation of Logistic model
    
    Takes an array of years and an array of corresponding SDG indicator values,
    checks if the observed and simulated arrays are of equal length
    or if they exceed the declared number of folds for K-folds,
    then run a logistic validation.

    Parameters
    ----------
    timeperiod : array-like
        Array of years where SDG indicators are recorded.
    value : array-like
        Array of SDG indicator values corresponding to the array of years.
    country : str
        Country name for raising warning.
    folds : int, optional
        Number of folds performed for K-fold validation. The default is 5.
    random_seed : int, optional
        Random seed for KFold function shuffling. The default is 1.

    Returns
    -------
    kfold_array : numpy array object
        An array of the averaged R^2, Percent-bias and Normalised root mean square error values for the K-fold validation.
        
    Warning
    -------
    Returns an array of NaNs if the checks fail or if the K-fold validation is otherwise unsuccessful.

    '''

    kfold_array = np.empty(3)*np.nan
    fold_data = None
    
    length_t = len(timeperiod)
    length_v = len(timeperiod)
    length_min = min(length_v, length_t)
    if length_t != length_v:
        # print("Length of inputs are not equal.")
        pass
    elif length_min < folds:
        # print("->{} dataset ({}) smaller than number of folds({})!".format(country,length_min,folds))
        pass
    
    else:
        fold_dataset = pd.concat([timeperiod,value], axis =1)
        fold_dataset.reset_index(drop=True)

        kfold = KFold(folds,shuffle=True,random_state = random_seed)
        splits = [i for i in kfold.split(fold_dataset)]
        # print(splits)
        
        fold_data = []

        
        for foldset in splits:
            train, test = foldset
            # print("Train:", train)
            # print("Test:", test)
            
            train_set = fold_dataset.iloc[train]
            try: 
                start, K, x_peak, r = calibration(train_set['TimePeriod'],train_set['Value'])
                test_set = fold_dataset.iloc[test]
    
                lf_results = [logistic(year, start, K, x_peak, r) for year in test_set['TimePeriod']]
                r2_value, pbias_value, nrmse_value = evaluation(test_set["Value"],lf_results)
                fold_data.append([r2_value, pbias_value, nrmse_value])
            except RuntimeError:
            # print("->No solution for fold logistic found for {}".format(country))
                pass
            

        fold_data = np.array(fold_data)
        kfold_array = np.nanmean(fold_data,axis = 0)
          
    return kfold_array

def main_analysis (goal_dataset, projection_year, timestep_list, num_folds = 5, k_fold_seed = 1):
    '''
    Perform Main Analysis
    
    Takes a dataframe of SDG indicator values by country and year,
    and performs a logistic regression, linear regression and logistic k-fold validation.
    
    Using the regression, the projected value of the SDG at 2030 is calculated per country
    and the evaluation of the regressions and validations on three criteria:
        R^2, PBIAS and NRMSE \n
    
    
    Parameters
    ----------
    goal_dataset : DataFrame object
        Dataframe object containing the SDG indicator values.
    projection_year : int
        Integer of the year for projection.
    timestep_list : array-like
        Array of timesteps for calulating simulated values from logistic and linear regression models of the SDG indicator.
    num_folds : int, optional
        Number of folds performed for K-fold validation. The default is 5.
    k_fold_seed : int, optional
        Random seed for KFold function shuffling. The default is 1.

    Returns
    -------
    results : dict
        A dictionary containing the following information for each country in the input dataframe:
            TimePeriod: list of years with recorded data、\n
            SimulationPeriod: timestep_list\n
            Value: list of observed values for SDG indicator\n
            Calibation: calibration parameters start, K, x_peak, r for logistic model
            
            Logistic Series: simulated vlaues using the logistic model at timessteps given by timestep_list\n
            Linear Series: simulated vlaues using the linear model at timessteps given by timestep_list
            
            Growth Rate: per capita growth rate of number of people affected by disaster (r) parameter for logistic model\n
            2030 Logistic: projected value of the SDG at 2030 from the logistic model\n
            2030 Linear: projected value of the SDG at 2030 from the linear model\n
                        
            Logistic/Linear/X-Fold R2, PIAS, NRMSE: Evaluation criteria valaues for model evaluations
    '''
    unique_countries = goal_dataset["GeoAreaName"].unique()
    results = {}
    
    for idx, country in enumerate(unique_countries):
        #for debug
        # print(country)
        
        #initialise values and dictionaries
        results[country] = {}
        x = goal_dataset.loc[goal_dataset.GeoAreaName == country, 'TimePeriod']
        y = goal_dataset.loc[goal_dataset.GeoAreaName == country, 'Value']
        
    
        try:
            #run logistic calibration
            start, K, x_peak, r = calibration(x,y)
            log_2030 = logistic(projection_year, start, K, x_peak, r)
            
            #generating logistic model results for datapoints
            log_eval = [logistic(year, start, K, x_peak, r) for year in x]
            #generating finaer time step
            log_list = [logistic(year, start, K, x_peak, r) for year in timestep_list]
            #performing Evaluation of regression
            log_r2, log_pbias, log_nrmse = evaluation(y,log_eval)
            
            # if math.isnan(log_pbias):
                # print("->{} has {} datapoints, not enough for regression".format(country,len(log_eval)))
                
        except RuntimeError:
            # print("->No solution for logistic found for {}".format(country))
            start = K = x_peak = r = log_2030 = log_list = log_pbias = log_pbias = log_nrmse = math.nan 
            
            pass
        
        
        try:
            #run linear regression
    
            slope, intercept, r_value, p_value, std_err = linregress(x,y)
            #generating linear model results
            lin_eval = [slope*year + intercept for year in x]
            #generating finer time step   
            lin_list = [slope*year + intercept for year in timestep_list]        
            
            lin_2030 = slope*projection_year +intercept
            #performing evaluation of regression
            lin_r2, lin_pbias, lin_nrmse = evaluation(y,lin_eval)
            
        except Exception:
            # print(traceback.format_exc())
            # print("\nLinear regression issue")
            pass
        
        #append values to results dictionary
        results[country]['TimePeriod'] = x.tolist()
        results[country]['SimulationPeriod'] = timestep_list
        results[country]['Value'] = y.tolist()
        results[country]['Calibration'] = (start, K, x_peak, r)
        
        results[country]['Logistic Series'] = log_list
        results[country]['Linear Series'] = lin_list
        
        results[country]['Growth Rate'] = r
        results[country]['2030 Logistic'] = log_2030
        results[country]['2030 Linear'] = lin_2030
        
        results[country]['Logistic evaluation R2'] = log_r2
        results[country]['Logistic evaluation PBIAS'] = log_pbias
        results[country]['Logistic evaluation NRMSE'] = log_nrmse
        
        results[country]['Linear evaluation R2'] = lin_r2
        results[country]['Linear evaluation PBIAS'] = lin_pbias
        results[country]['Linear evaluation NRMSE'] = lin_nrmse
        
        fold_results = fold_logistic(x,y,country, num_folds, k_fold_seed)
        results[country]['{}-Fold validation mean R2'.format(num_folds)] = fold_results[0]
        results[country]['{}-Fold validation mean PBIAS'.format(num_folds)] = fold_results[1]
        results[country]['{}-Fold validation mean NRMSE'.format(num_folds)] = fold_results[2]
    
    return results


def plot_trendlines (country_list, results_dict):
    '''
    Takes a list of countries and a dictionary of main_analysis results, and creates:
        a scatter plot of their observed values
        a line plot of their logistic regression
        a line plot of their linear regression

    Parameters
    ----------
    country_list : array
        Array of keys in results_dict corresponding to the countries to be plotted
    results_dict : dict
        Dictionary of results from main_analysis containing the lists of observed values and simulated values for logistic and linear models per country

    Returns
    -------
    fig : matplotlib figure object
        A figure containing the observed, logistic-simulated and linear-simulated plots of the countries in country_list.

    '''
    #initialise plot
    fig = plt.figure(figsize = (6,4))
    plt.xlabel("Year", fontsize=10)  
    plt.ylabel("Number of people affected by disaster", fontsize=10)
    
    # list of tableau colours 
    tableau10 = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    
    # print(country_list)
    for idx, country in enumerate(country_list):    
        #create variables for plotting
        timeperiod = results_dict[country]['TimePeriod']
        simperiod = results_dict[country]['SimulationPeriod']
        obs = results_dict[country]['Value']
        log_sim = results_dict[country]['Logistic Series']
        lin_sim = results_dict[country]['Linear Series']
        
        #plot functions
        plt.scatter(timeperiod, obs, color = tableau10[idx], label="{}: Observed".format(country),  )
        plt.plot(simperiod, log_sim, color = tableau10[idx], label="{}: Logistic Simulated".format(country), linestyle = "dashed" )
        plt.plot(simperiod, lin_sim, color = tableau10[idx], label="{}: Linear Simulated".format(country) )
    
    # plt.xlim(min(timeperiod),max(timeperiod))
    plt.xticks(np.arange(2004,2021,2))
    plt.ylim(min(min(obs),min(lin_sim),min(log_sim)))
    
    # include legend
    ax = fig.gca()
    handles, labels = ax.get_legend_handles_labels()
    # sort both labels and handles by labels
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    ax.legend(handles, labels)
    
    plt.tight_layout()
    plt.show()
       
    return fig 

def plot_dotplot (df, country_column, growth_column, proj_column):
    '''
    Takes a dataframe of countries and their main_analysis,
    and the column names for Country, Growth Rates and Projected 2030 values,
    and creates:
        an ordered dot plot indicating the growth rate with the colour indicating the projected values in 2030.


    Parameters
    ----------
    df : pandas DataFrame object
        A DataFrame containing the main-analysis values of countries in the SDG dataset
    country_column : str
        Column name of country names in DataFrame.
    growth_column : str
        Column names of growth rates in DataFrame.
    proj_column : str
        Column names of projected 2030 values in DataFrame.

    Returns
    -------
    fig : matplotlib figure object
        A figure containing an ordered dot plot indicating the growth rate with the colour indicating the projected values in 2030.

    '''
    fig = plt.figure(figsize = (6,5), dpi=300)  
    plt.axvline(0, 0, 250, color = "slategrey", alpha = 0.5, label = "0 growth rate") 
    plt.scatter(df[growth_column], df[country_column], c=df[proj_column], cmap = 'rainbow' ,)
    plt.colorbar(pad = 0.02, label="Projected Number of people affected by disaster in 2030")
    
    
    plt.xlabel("Growth Rate (per capita growth rate of number of people affected by disaster)", fontsize = 10)
    plt.xlim(-10.5,10.5)
    plt.tight_layout()
    plt.show()
    return fig