# -*- coding: utf-8 -*-
"""
Read and plot S-MAP L-band soil moisture data as provided in daily netcdfs by VanderSat

Developer: Rogier Westerhoff, GNS Science.
Data produced by: VanderSat / Planet, The Netherlands

v001: test for reading one file
v002: plots daily maps and mean maps, option of plotting all (daily) maps can be switched on or off
v003, 24 Feb 2022: plotting time series of a specific coordinate
v004, 24 Feb 2022: re-structuring for efficiency
v005: 2 March:
      - further re-writing for efficiency;
      - Changed data folder location
Spyder Editor
v006: 6 May 2022:
PyCharm editor
    - plot time series of coordinate
"""
# PyCharm: Press Shift+F10 to run script

import netCDF4 as nc
import glob
import matplotlib.pyplot as plt
#import matplotlib.dates as mdates

import numpy as np
import datetime
import time
import pandas as pd
import math

# +++++++++++ INPUT VARIABLES +++++++++++++++++++++++
plot_maps = False
plot_all_maps = False
plot_time_series = True

save_my_figs = True

lat_point = -35.5
lon_point = 174.0
start_date = datetime.datetime(2016,6,1)
end_date = datetime.datetime(2021,5,31) # year, month, day

# ++++++++ MY FUNCTIONS +++++++

from libs.modules.utils import indexContainingSubstring, closestNode

def plotSmMaps(data,my_extent,labelstr,defaultScale,save_my_fig):
        
    # minVal = np.percentile(data.compressed(),5) #min_value = np.min(sm)
    # maxVal = np.percentile(data.compressed(),95) #max_value = np.max(sm)
    
    if defaultScale: # UGLY!! TRY args and kwargs (but having trouble passing kwargs as variable names)
        minVal = 0
        maxVal = 0.5
    else:
        maxVal = np.max(data)
        minVal = np.min(data)
    
    lon_min = my_extent[0]
    lon_max = my_extent[1]
    lat_min = my_extent[2]
    lat_max = my_extent[3]
    
    fig, ax = plt.subplots(figsize=(6,6))
    
    ax.set_aspect('equal')
    smplot = plt.imshow(data,cmap=plt.cm.terrain_r, interpolation='none',\
    #       extent=[lon_min,lon_max,lat_min,lat_max],vmin=0.15,vmax=0.45)
           extent=my_extent,vmin=minVal,vmax=maxVal)
    plt.text(lon_max-0.05,lat_max-0.1, labelstr,ha='right') #, bbox={'facecolor': 'white', 'pad': 10})
    
    cbar = fig.colorbar(smplot,ax=ax,anchor=(0, 0.3), shrink=0.8)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('m $^3$ m $^{-3}$', rotation=270, va='top')
    if save_my_fig:
        plt.savefig(r'files/outputs/sm_'+ labelstr+'.png',dpi=300)
    else:
        plt.show()

# +++++++ END OF FUNCTIONS +++++

#from datetime import datetime
    
t = time.time()

#path = r'I:\GroundWater\Research\NIWA_NationalHydrologyProgram\Data\SoilMoistureVanderSat\NorthlandOutput\SMAP_LN\'
#print(glob.glob(r"I:\GroundWater\Research\NIWA_NationalHydrologyProgram\Data\SoilMoistureVanderSat\NorthlandOutput\SMAP_LN\*.nc")) 
file_list = sorted(glob.glob(r"i:\GroundWater\Research\NIWA_NationalHydrologyProgram\Data\SoilMoistureVanderSat\NetCDF_files\NorthlandNcFiles\SM-SMAP*.nc"))

# https://www.w3schools.com/python/python_datetime.asp

a = start_date.strftime('%Y%m%d')
start_date_str = (a[:4] + r'-' + a[4:6]+r'-'+a[6:8])

a = end_date.strftime('%Y%m%d')
end_date_str = (a[:4] + r'-' + a[4:6]+r'-'+a[6:8])

duration_days = (end_date - start_date).days + 1
print(str(duration_days)+r' day period')

index1 = indexContainingSubstring(file_list,start_date_str)
index2 = indexContainingSubstring(file_list,end_date_str)

# build a pandas dataframe for dates and soil moisture values
column_names = ["date", "soil_moisture"]
df = pd.DataFrame(columns = column_names)

i = 0
# start loop
for ifile in range(index1,index2+1):
 
    fn = file_list[ifile]
    ds = nc.Dataset(fn)
    sm = ds['SM-SMAP-L-DESC_V4.0_100'][0,:,:]
    sm = np.ma.array(sm) # ma because of the masked array
    date_label = ds.datetime[0:10]  
    
    if ifile == index1:
       print(r'header first file: '+date_label)
       lat = ds['lat'][:]
       lon = ds['lon'][:]
       
    if ifile == index2:
       print(r'header last file: '+ds.datetime[0:10])
    
    if plot_maps:       
       if ifile == index1:
          lon_min = float(ds.__dict__['lon_min'])
          lon_max = float(ds.__dict__['lon_max'])
          lat_min = float(ds.__dict__['lat_min'])
          lat_max = float(ds.__dict__['lat_max'])
          my_extent=[lon_min,lon_max,lat_min,lat_max]
          smMapSeries = sm
       else:
          smMapSeries = np.ma.dstack((smMapSeries,sm)) 
        
       if plot_all_maps: # plot all (often, not always, daily) maps
          plotSmMaps(sm,my_extent,date_label,True,save_my_figs)
        
       if ifile == index2: # plot map with mean at last date
          sm_mean = np.ma.mean(smMapSeries, axis=(2))
          label_str = r'mean_' + start_date.strftime('%Y%m%d') + r'_' + end_date.strftime('%Y%m%d')
          plotSmMaps(sm_mean,my_extent,label_str,True,save_my_figs)
    
    if plot_time_series:
       index_lat = closestNode(lat_point, lat) # only used in timeseries
       index_lon = closestNode(lon_point, lon) # only used in timeseries
       sm_q = sm[index_lon,index_lat]
       if (bool(sm_q)):   
           date_time_obj = datetime.datetime.strptime(ds.datetime, '%Y-%m-%d %H:%M:%S')
           df.loc[i] = [date_time_obj,sm_q] # put data in dataframe
       else:
           sm_q = np.nan
           
    i += 1   

if plot_time_series:
    df = df.set_index('date') # https://www.dataquest.io/blog/tutorial-time-series-analysis-with-pandas/

    my_fontsize = 14
    year_size = 365 #approx 5 years of daily data
    df['soil_moisture'].plot(marker='.', ms = 8, alpha=1, linestyle='None',
      figsize=(5*(math.ceil(df.size/year_size)),5), fontsize=my_fontsize, grid = True)
    # sm_av = movingaverage(df['soil_moisture'], 50) # testing
    plt.title('SMAP timeseries for queried coordinate', fontsize=my_fontsize)
    plt.xlabel('', fontsize=my_fontsize)
    plt.ylabel('Soil moisture (m$^3$/ m$^3$)', fontsize=my_fontsize)  
    plt.tight_layout()

    if save_my_figs:
        plt.savefig(r'files/outputs/ts_sm_'+str(lon_point)+'_'+str(lat_point)+'.png',dpi=300)
    else:
        plt.show()
   

elapsed = time.time() - t
print(r'run time: '+ str(round(elapsed)/60) + r' minutes')           
