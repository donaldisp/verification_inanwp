## This script is used to do verification of InaNWP forecast with Synop observation data
# Created by Donaldi Permana
# 08/09/2022
# Email : donaldi.sp@gmail.com; donaldi.permana@bmkg.go.id
# Copyright : BMKG

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from math import sqrt
import os
import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", RuntimeWarning)

##========= reading observation file ========
bmkgsoft_file = '96745_databmkgsoft_2021.csv'
df = pd.read_csv(bmkgsoft_file, sep=',')
#print(df)

# drop rows with duplicate column "ID_STATION" and "DATE_TIME"
df.drop_duplicates(subset=['ID_STATION', 'DATE_TIME'],inplace=True)
# create new column 'DateTime'
df['DateTime'] = df['DATE_TIME']
# set 'DateTime' as datetime format
df['DateTime'] = pd.to_datetime(df['DateTime'], format="%d/%m/%Y %H:%M")
# sort by 'DateTime'
df = df.sort_values(by='DateTime')
# set 'DateTime' as index
df.set_index('DateTime', inplace = True)

#print(df.info())
#print(df)

# QC data observation (range check)
# QC for mslp
df['mslp'] = df['SEA_LEVEL_PRESSURE'].astype('float64') # mean sea level pressure
df.loc[df['mslp'] > 1025, 'mslp'] = np.nan

# QC for surfp
df['surfp'] = df['SURFACE_PRESSURE'].astype('float64') # surface pressure
df.loc[df['surfp'] > 1025, 'surfp'] = np.nan

# QC for t
df['t'] = df['DRY_TEMP'].astype('float64') # temperature
df.loc[df['t'] > 50, 't'] = np.nan
df.loc[df['t'] < 10, 't'] = np.nan

# QC for td
df['td'] = df['DEW_POINT'].astype('float64') # temperature dewpoint
df.loc[df['td'] > 50, 'td'] = np.nan
df.loc[df['td'] < 10, 'td'] = np.nan

# QC for rh
#df['rh'] = 100*(np.exp((17.625*df['td'])/(243.04+df['td']))/np.exp((17.625*df['t'])/(243.04+df['t'])))
df['rh'] = df['REL_HUM'].astype('float64') # temperature dewpoint
# set rh > 100 to 100
df.loc[df['rh'] > 100, 'rh'] = 100
df.loc[df['rh'] < 0, 'rh'] = np.nan

# QC for dd
df['dd'] = df['WIND_DIR'].astype('float64') # wind direction
df.loc[df['dd'] > 365, 'dd'] = np.nan
df.loc[df['dd'] < 0, 'dd'] = np.nan

# QC for ff
df['ff'] = df['WIND_SPEED'].astype('float64') # wind speed
df.loc[df['ff'] > 20, 'ff'] = np.nan # set wind speed more than 20 m/s to nan

# QC for rr3
df['rr3'] = df['RAINFALL_3H'].astype('float64') # precipitation
df.loc[df['rr3'] > 100, 'rr3'] = np.nan # set rain3hour more than 100 mm to nan

# copy var df to df_obs
df_obs = df.copy()

##==============================

##===== plotting part - observation ==

# set variable to be plotted
var = 't' # option = t, tc, rh, dd, ff, rr3

plt.rcParams.update({'font.size': 14})
plt.figure(figsize=(12,8), dpi=75)

# plot selected variable
df[var].plot()
plt.ylabel('deg C')
#plt.show()

# plot boxplot by month
fig, ax = plt.subplots()
values, labels, jitters = [], [], []

grouped = df.groupby(df.index.month)

position = 1
for key, group in grouped: # looping
    values.append(group[var].values)
    labels.append(key)
    jitters.append(np.random.normal(position, 0.04, group[var].values.shape[0]))
    position += 1
    
ax.boxplot(values, labels=labels, showfliers=False)
plt.setp(ax.get_xticklabels(), rotation=90)
plt.ylabel('deg C')

for x, val in zip(jitters, values):
    ax.scatter(x, val, alpha=0.4, linewidth=0)

#plt.show()

##=====================

##======= defining functions =========
#If you want errors ranging from 0 to 180 you can use the following function :
def wdir_diff(wd1,wd2):
    wd1 = np.array(wd1)
    wd2 = np.array(wd2)
    diff1 = (wd1 - wd2)% 360
    diff2 = (wd2 - wd1)% 360
    res = np.minimum(diff1, diff2)
    return res

def wdir_rmse(predictions, targets):
    return np.sqrt((wdir_diff(predictions,targets) ** 2).mean())

##=====================================

##==== reading the InaNWP forecast files =====

# get filenames in fcst folder
fcstnames = os.listdir('fcst')
#print(fcstnames)

df_fcst_all = []
for fname in fcstnames:
    inittime = fname.split('_')[1]
    print('Reading forecast for initial '+inittime+' ...')
        
    df_fcst = pd.read_csv('fcst/'+fname, sep=',')
    df_fcst.drop(df_fcst.columns[0], axis=1, inplace=True)
    df_fcst['datetime'] = pd.to_datetime(df_fcst['datetime'], format="%Y-%m-%d %H:%M:%S")
    df_fcst.set_index('datetime', inplace = True)
    
    #calculate variable wd10 from u10 and v10
    df_fcst['wd10'] = np.mod(180+np.rad2deg(np.arctan2(df_fcst['u10'], df_fcst['v10'])),360)    
    
    #print(df_fcst)

    # append dataframe to df_fcst_all
    df_fcst_all.append(df_fcst)
    #print(df_fcst)

##======================================

##======== main script for verification ================##

# variables to verify
variables = ['t2m','rh2','td2m','mslp','ws10','wd10','rain03']
str_label = {'t2m':' T 2m (degree C)', \
             'rh2':' RH 2m (%)', \
             'td2m':' TD 2m (degree C)', \
             'mslp':' MSLP (mb)', \
             'ws10':' wind speed (m/s)', \
             'wd10':' wind dir (degree)', \
             'rain03':' 3-hour prec (mm)'}
# correspond variables in fcst to obs
obs_var = {'t2m':'t', 'rh2':'rh', 'td2m':'td','mslp':'mslp','ws10':'ff','wd10':'dd','rain03':'rr3'}

dirout = 'output1/'
# create output directory if not exist
if not os.path.exists(dirout):
    os.makedirs(dirout)

# set font size in plots
plt.rcParams.update({'font.size': 7})
n_forecast_day = 3 # 3 days
interval = 3 # 3 hourly

stats = []
stats_columns = []
stats_item = []

#add column fcst lead as first column
stats_columns = ['fcst lead']
stats = [np.arange(interval,interval*len(df_fcst_all[0]),interval)]

for var in variables:
    corrcoef_all = []
    rmse_all = []
    n_all = []
    for lead in range(len(df_fcst_all[0])):
        str_lead = str(round(lead*interval,0))
        if lead*interval < 10:
            str_lead = '0'+str_lead
        print('Verifying '+ var + ' for lead +'+str_lead+' hour ...')
        fcst_times = [df.index[lead] for df in df_fcst_all] # iterative dataframe
        fcst = [df[var].iloc[lead] for df in df_fcst_all]
        obs = df_obs[obs_var[var]].loc[fcst_times].to_numpy()
        #print(obs)
              
        # calculate corr and rmse for all days fcst (depend on the length of not null obs)                
        idxx = ~np.isnan(obs)
        obs_all = obs[idxx]
        fcst_all = np.array(fcst)[idxx]

        if var == 'wdir':
            error = round(wdir_rmse(obs_all, fcst_all),2)
        else:
            MSE = np.square(np.subtract(obs_all,fcst_all)).mean() 
            RMSE = math.sqrt(MSE)
            error = round(RMSE,2)
        corrcoef = round(np.corrcoef(obs_all, fcst_all)[0][1],2)
        n = len(obs_all)
        #if corrcoef == 1 or corrcoef == -1 or n < 3:
        #    error = np.nan
        #    corrcoef = np.nan
        #    n = np.nan
        corrcoef_all.append(corrcoef)
        rmse_all.append(error)
        n_all.append(n)

        #--- plot fcst vs obs
        # fig, ax = plt.subplots()
        # ax.plot(fcst_times,obs, 'bo-',label='obs')
        # ax.plot(fcst_times,fcst, 'r*-',label='fcst')
        # ax.legend(loc='best')
        # title = 'Forecast lead +'+str_lead+' hour'
        # title = title + ' (r = '+str(corrcoef)+', rmse = '+str(error)+', n = '+str(n)+')'
        # ax.set_title(title)
        # ax.set_xlim(np.min(fcst_times), np.max(fcst_times))
        # ax.set_ylabel(str_label[var])
        # fig.savefig(dirout+ var+'_'+str_lead+'.png', format='png', dpi=90, bbox_inches='tight')
        #plt.show()

    stats_columns.append('corr_'+var)
    stats_columns.append('rmse_'+var)
    stats_columns.append('n_'+var)
    print(len(stats_columns))

    stats.append(corrcoef_all[1:])
    stats.append(rmse_all[1:])
    stats.append(n_all[1:])
    print(len(stats))
    
    #-- plot corr-coef
    fig, ax = plt.subplots(2, figsize=(10,5))
    ax[0].plot(np.arange(3,75,3),corrcoef_all[1:], 'bo-',label=var)
    ax[0].set_ylabel('Corr. coef')
    ax[0].legend(loc='best')
    ax[0].set_xticks(np.arange(3,75,3))
    ax[0].set_xticklabels(np.arange(3,75,3))
    #ax[0].set_xlabel('fcst lead (hour)')
    ax[0].set_title('verification '+str_label[var])
    #-- plot rmse
    ax[1].plot(np.arange(3,75,3),rmse_all[1:], 'ro-',label=var)
    ax[1].set_ylabel('rmse')
    ax[1].legend(loc='best')
    ax[1].set_xticks(np.arange(3,75,3))
    ax[1].set_xticklabels(np.arange(3,75,3))
    ax[1].set_xlabel('forecast lead (hour)')
    #ax[1].set_title(str_label[var])
     
    fig.savefig(dirout+ var+'_corr_rmse.png', format='png', dpi=500, bbox_inches='tight')
    #plt.show()
    
    # plotting boxplot for corr
    fig, ax = plt.subplots(figsize=(10,5))
    values, labels, jitters = [], [], []
    for day in range(n_forecast_day):
        day += 1
        vals = corrcoef_all[int((day-1)*(24/interval))+1:int(day*(24/interval))+1]
        vals_notna = np.array(vals)[~np.isnan(vals)]
        #break
        values.append(vals_notna)
        labels.append('corr '+var+' day-'+ str(day) + '\n (mean = '+\
                        str(round(np.mean(vals_notna),2))+') \n (median = '+\
                        str(round(np.median(vals_notna),2))+') \n (n = '+\
                        str(len(vals_notna))+')')
        jitters.append(np.random.normal(day, 0.04, vals_notna.shape[0]))
    # plot boxplot
    ax.boxplot(values, labels=labels, showfliers=True)
    # plot scatter plot in boxplot area with jitter
    colors = ['r', 'g', 'b', 'y']
    for x, val, c in zip(jitters, values, colors):
        ax.scatter(x, val, alpha=0.4, color=c, linewidth=0)

    ax.set_ylabel('corr')
    ax.set_title('verification' + str_label[var])

    fig.savefig(dirout+ var +'_boxplot_corr.png', format='png', dpi=500, bbox_inches='tight')
    #plt.show()
    
    # plotting boxplots for rmse
    fig, ax = plt.subplots(figsize=(10,5))
    values, labels, jitters = [], [], []
    for day in range(n_forecast_day):
        day += 1
        vals = rmse_all[int((day-1)*(24/interval))+1:int(day*(24/interval))+1]
        vals_notna = np.array(vals)[~np.isnan(vals)]
        #break
        values.append(vals_notna)
        labels.append('rmse '+var+' day-'+ str(day) + '\n (mean = '+\
                        str(round(np.mean(vals_notna),2))+') \n (median = '+\
                        str(round(np.median(vals_notna),2))+') \n (n = '+\
                        str(len(vals_notna))+')')
        jitters.append(np.random.normal(day, 0.04, vals_notna.shape[0]))
    # plot boxplot
    ax.boxplot(values, labels=labels, showfliers=True)
    # plot scatter plot in boxplot area with jitter
    colors = ['r', 'g', 'b', 'y']
    for x, val, c in zip(jitters, values, colors):
        ax.scatter(x, val, alpha=0.4, color=c, linewidth=0)

    ax.set_ylabel('rmse')
    ax.set_title('verification' + str_label[var])

    fig.savefig(dirout+ var +'_boxplot_rmse.png', format='png', dpi=500, bbox_inches='tight')
    #plt.show()

ds = pd.DataFrame(np.transpose(stats), columns=stats_columns)
#ds.info()
# save all results to csv file
ds.to_csv(dirout+ 'stats.csv',index=False)