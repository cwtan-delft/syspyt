    # -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 16:26:38 2021

@author: tanch
"""
#####################
##Assignment B
#####################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cProfile 
import pstats 
import time

from assignment_B_functions import main_analysis, plot_trendlines, plot_dotplot
from assignment_B_window import window

import warnings
warnings.filterwarnings("ignore")
#%% 
##################################
# Part7: Start profiling 
##################################s

# pr=cProfile.Profile() 
# pr.enable()
tstart = time.time()  

#%% Import Data

##################################
# Part1: Import Data into Python with Pandas
##################################

# SDG 13.1.1
goal13 = pd.read_excel("Goal13.xlsx","data")
metric = 'Number of people affected by disaster (number)'

t1 = time.time() 
##################################
# Part2: Data Selection
##################################

#Data selection - 'Number of people affected by disaster (number)'
goal13_affected = goal13[goal13["SeriesDescription"]==metric]

# create pivot table of country vs year for number of people affected by disaster
pivot13 = goal13_affected.pivot(index=list(goal13_affected)[0:7], columns="TimePeriod", values="Value")

# Population 
world_pop = pd.read_excel("WPP2019_POP_F01_1_TOTAL_POPULATION_BOTH_SEXES.xlsx","ESTIMATES", header = 16, index_col=0)
country_pop = world_pop.loc[world_pop.Type == "Country/Area"]
#top 30 largest countries
country_pop_t30 = country_pop.sort_values("2020", ascending= False).head(30)

t2 = time.time() 
#%% Sanity Check

##################################
# Part3: Make sanity checks and print them to console
##################################

sanity = []
#counting non-missing values per country
country_counts = goal13_affected["GeoAreaCode"].value_counts()

unique_countries = goal13_affected["GeoAreaCode"].unique()
for country in unique_countries:
    country_name = goal13_affected.loc[goal13_affected.GeoAreaCode == country, 'GeoAreaName'].unique()[0]
    #no of values per country
    country_count = goal13_affected.loc[goal13_affected.GeoAreaCode == country, 'GeoAreaName'].count()
    #min
    country_min = goal13_affected.loc[goal13_affected.GeoAreaCode == country, 'Value'].min()
    #max
    country_max = goal13_affected.loc[goal13_affected.GeoAreaCode == country, 'Value'].max()
    #mean
    country_mean = np.mean(goal13_affected.loc[goal13_affected.GeoAreaCode == country, 'Value'])
    
    sanity.append([country, country_name, country_count,country_min,country_max,country_mean])

sanity_df = pd.DataFrame(sanity, columns=["GeoAreaCode","GeoAreaName","Observations","Minimum","Maximum","Mean"])
sanity_df.set_index( 'GeoAreaCode',inplace=True)

print("Sanity Check")
print("----------------------------------------------")
print("The maximum number of non-missing values is {} ".format(sanity_df.Observations.max()))
print("The minimum number of non-missing values is {} ".format(sanity_df.Observations.min()))
print("The mean number of non-missing values is {:.2f} \n".format(sanity_df.Observations.mean()))


#print countries with the maximum amount of datapoints
# print([(k,v) for k,v in country_counts.items() if v == country_counts.max()])
#sort sanity dataframe by max number of observations
# sanity_by_max = sanity_df[sanity_df.Observations >=10].sort_values("Maximum", ascending = False)

t3 = time.time() 
#%% Main Analysis

pr=cProfile.Profile() 
pr.enable()

##################################
# Part4: Perform the main analysis
##################################

def func():
    num_folds = 5
    ##################################
    '''
    comment this out to disable GUI
    '''
    # if __name__ == "__main__":
    #     num_folds = window()
    ##################################
    
    #set variables for logistic evaluation and validation here
    projection_year = 2030
    k_fold_seed = 1
    
    
    years_list = np.linspace(2005,2020,150)     
    
    results = main_analysis(goal13_affected, projection_year, years_list, num_folds, k_fold_seed)
    
    #create a dataframe from results dictionary
    results_df = pd.DataFrame.from_dict(results,orient='index')
    results_df.index.name = 'GeoAreaName'

t4 = time.time()   

pr.disable() 
ps=pstats.Stats(pr).strip_dirs().sort_stats('cumulative')
ps.print_stats(20) 
ps.print_stats('script_to_search')
ps.print_callers(20)

# %reload_ext snakeviz
# %snakeviz func()

#%% plot for 2 countries

##################################
# Part5A: Select 2 contrasting countries and display their trends
##################################

country_plot =  ["Niger", "Sri Lanka"]

fig_2_countries = plot_trendlines(country_plot, results)
fig_name = "{}_vs_{}_timeseries.png".format(country_plot[0],country_plot[1])
fig_2_countries.savefig(fig_name,bbox_inches="tight")
plt.close(fig_2_countries)

t5a = time.time() 
#%% ordered dot plot of 30 most polulous countries

##################################
# Part5B: display an ordered dot plot of 30 most poplulous countries
#           indicating the growth rate with the colour indicating the projected value in 2030
##################################
'''['Nigeria', 'Democratic Republic of the Congo', 'Germany', 'United Kingdom', 'Italy', 'Spain'] not in index'''

#preparation of dataset
results_30_df = pd.merge(results_df,country_pop_t30[["Region, subregion, country or area *","2020"]], left_index=True, right_on = "Region, subregion, country or area *", how="right")
results_30_df.sort_values("Growth Rate", ascending = False, inplace=True)
results_30_df.rename(columns = {"Region, subregion, country or area *": 'GeoAreaName', "2020":"2020 Population"}, inplace=True)
results_30_df.set_index('GeoAreaName')
plot_df = results_30_df[["GeoAreaName","2020 Population", "2030 Logistic", "2030 Linear", "Growth Rate"]]
plot_df.dropna(axis = 0, inplace = True)

#plot
growthplot = plot_dotplot(plot_df, "GeoAreaName", "Growth Rate", "2030 Logistic")
growthplot.savefig("dot.png")
plt.close(growthplot)

t5b = time.time() 
#%% create csv file

##################################
# Part6: Export Main results to a csv file
##################################
csv_name = "SDG13_simulation.csv"

results_df = pd.DataFrame.from_dict(results,orient='index')
results_df.index.name = 'GeoAreaName'
csv_indexes = pd.DataFrame(index=pivot13.index)
csv_data = pd.merge(csv_indexes, results_df[list(results_df)[-11:]], left_index=True, right_index=True, how='left')
csv_data.to_csv(csv_name)

t6 = time.time() 
#%%
##################################
# Additional code for presentation
##################################

#%% sanity check only for 30 most populous countries
# sanity_30 = pd.merge(sanity_df, country_pop_t30[["Region, subregion, country or area *"]],left_on="GeoAreaName", right_on="Region, subregion, country or area *", how= "right")

#%% plot for log evaluation scores
# results_drop = results_df.dropna(axis = 0)
# columns = ['Logistic evaluation R2','Logistic evaluation PBIAS', 'Logistic evaluation NRMSE']
# fig, ax = plt.subplots(1,3,dpi=300)

# for idx,col in enumerate(columns):
#     ax[idx].boxplot(results_drop[col],showfliers=False)
#     ax[idx].set_xlabel(col)
# fig.suptitle ("Logistic Evaluation")
# plt.tight_layout()
# fig.savefig("log_boxplot.png")
# plt.show()
# plt.close()

#%% plot for linear evaluation scores
# columns = ['Linear evaluation R2','Linear evaluation PBIAS', 'Linear evaluation NRMSE']
# fig, ax = plt.subplots(1,3,dpi=300)

# for idx,col in enumerate(columns):
#     ax[idx].boxplot(results_drop[col],showfliers=False)
#     ax[idx].set_xlabel(col)
# plt.suptitle ("Linear Evaluation")
# plt.tight_layout()
# fig.savefig("linear_boxplot.png")
# plt.show()
# plt.close()


#%% plot for linear evaluation scores
# columns = ['{}-Fold validation mean R2'.format(num_folds),'{}-Fold validation mean PBIAS'.format(num_folds), '{}-Fold validation mean NRMSE'.format(num_folds)]
# fig, ax = plt.subplots(1,3,figsize= (8,4), dpi=300)

# for idx,col in enumerate(columns):
#     ax[idx].boxplot(results_drop[col],showfliers=False)
#     ax[idx].set_xlabel(col)
# fig.suptitle ("K-Fold Validation")
# plt.tight_layout()
# fig.savefig("kfold_boxplot.png")
# plt.show()
# plt.close()

#%% Plotting inidividual observed vs simulated trends for top 30 most populous countries

# #list of countries for plotting
# country_plot = country_pop_t30["Region, subregion, country or area *"]

# #plot
# for country in country_plot:
#     if country in results.keys():
#         fig = plot_obs_sim(country,results)
#         fig_name = "obsevered_vs_simulated_trend_{}.png".format(country)
#         fig.savefig(fig_name,bbox_inches="tight")
#         plt.close(fig)
#     else:
#         print("{}: missing plot".format(country))


#%% profiling end code

##################################
# Part7: End profiling 
##################################
# pr.disable() 
# ps=pstats.Stats(pr).strip_dirs().sort_stats('cumulative')
# ps.print_stats(10) 
# ps.print_stats('script_to_search')
# ps.print_callers()

# tend = time.time()  
# print('Done script in %5.2f s\n'% (tend - tstart))

time1 = t1- tstart
time2 = t2- t1
time3 = t3 - t2
time4 = t4-t3
time5a = t5a - t4
time5b = t5b - t5a
time6 = t6 - t5b

timefig, ax= plt.subplots(dpi = 300)
x = ["Part1: Import data", "Part2: Data selection", "Part3: Make sanity checks",
     "Part4: Perform the main analysis", "Part5A: Select 2 contrasting countries and display their trends",
     "Part5B: display an ordered dot plot of 30 most poplulous countries",
     "Part6: Export Main results to a csv file"]
y=[time1, time2, time3, time4, time5a, time5b, time6]

hbars = ax.barh(x,y)
plt.xlabel("Time (s)")
plt.xlim(0,150)
ax.bar_label(hbars, fmt='%.2f')

plt.show()
timefig.savefig("timing for each part.png", bbox_inches="tight")
plt.close(timefig)