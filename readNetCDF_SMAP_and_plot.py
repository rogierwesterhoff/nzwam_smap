# -*- coding: utf-8 -*-
"""
Read and plot SMAP L-band soil moisture data as provided in daily netcdfs by VanderSat

Developer: Rogier Westerhoff, GNS Science.
Data produced by: VanderSat / Planet, The Netherlands

v001: test for reading one file
v002: plots daily maps and mean maps, option of plotting all (daily) maps can be switched on or off
v003, 24 Feb 2022: plotting time series of a specific coordinate
v004, 24 Feb 2022: re-structuring for efficiency
v005: 2 March:
   - further re-writing for efficiency;
   - Changed data folder location
(v001 to v005 were developed in the Spyder Editor)
v006: 6 May 2022:
    -   developed in PyCharm environment
    -   plot time series of coordinate
    -   added smoothing average line
"""
# PyCharm: Press Shift+F10 to run script

import os
import netCDF4 as nc
import glob
import matplotlib.pyplot as plt
# import matplotlib.dates as mdates

import numpy as np
import datetime
import time
import pandas as pd
import math

# +++++++++++ INPUT VARIABLES +++++++++++++++++++++++
plot_maps = True
plot_all_maps = False
plot_time_series = False

save_my_figs = True

lat_point = -35.5
lon_point = 174.0
start_date = datetime.datetime(2016, 6, 1)
end_date = datetime.datetime(2016, 6, 30)  # year, month, day

# ++++++++ MY FUNCTIONS +++++++

from libs.modules.utils import indexContainingSubstring, closestNode, movingaverage
from libs.modules.my_methods import plotSmMaps

# +++++++ END OF FUNCTIONS +++++

work_dir = os.getcwd()
# EXAMPLE: my_path = os.path.join(work_dir, r'files\dataframe\')
# from datetime import datetime

t = time.time()

my_path = r'i:GroundWater\Research\NIWA_NationalHydrologyProgram\Data\SoilMoistureVanderSat\SmapData\NorthlandNcFiles'
# print(glob.glob(r"I:\GroundWater\Research\NIWA_NationalHydrologyProgram\Data\SoilMoistureVanderSat\NorthlandOutput\SMAP_LN\*.nc"))
file_list = sorted(glob.glob(os.path.join(my_path, r'SM-SMAP*.nc')))

# https://www.w3schools.com/python/python_datetime.asp

a = start_date.strftime('%Y%m%d')
start_date_str = (a[:4] + r'-' + a[4:6] + r'-' + a[6:8])

a = end_date.strftime('%Y%m%d')
end_date_str = (a[:4] + r'-' + a[4:6] + r'-' + a[6:8])

duration_days = (end_date - start_date).days + 1
print(str(duration_days) + r' day period')

index1 = indexContainingSubstring(file_list, start_date_str)
index2 = indexContainingSubstring(file_list, end_date_str)

# define an empty pandas dataframe for dates and soil moisture values, smoothing average and possibly more later
column_names = ["date", "soil_moisture"]
df = pd.DataFrame(columns=column_names)

i = 0
# start loop
for ifile in range(index1, index2 + 1):

    fn = file_list[ifile]
    ds = nc.Dataset(fn)
    date_label = ds.datetime[0:10]

    if ifile == index1:
        print(r'header first file: ' + date_label)

    if ifile == index2:
        print(r'header last file: ' + ds.datetime[0:10])

    if plot_maps: # assumes that all files have the same array size
        # sm = ds['SM-SMAP-L-DESC_V4.0_100'][0, :, :]
        sm = np.ma.array(ds['SM-SMAP-L-DESC_V4.0_100'][0, :, :]) # ma because of the masked array
        lon_min = float(ds.__dict__['lon_min'])
        lon_max = float(ds.__dict__['lon_max'])
        lat_min = float(ds.__dict__['lat_min'])
        lat_max = float(ds.__dict__['lat_max'])
        my_extent = [lon_min, lon_max, lat_min, lat_max]
        if ifile == index1:
            smMapSeries = sm
        else:
            smMapSeries = np.ma.dstack((smMapSeries, sm))

        if plot_all_maps:  # plot all (daily) maps
            plotSmMaps(sm, my_extent, date_label, True, save_my_figs)

        if ifile == index2:  # plot map with mean at last date
            sm_mean = np.ma.mean(smMapSeries, axis=(2))
            label_str = r'mean_' + start_date.strftime('%Y%m%d') + r'_' + end_date.strftime('%Y%m%d')
            plotSmMaps(sm_mean, my_extent, label_str, True, save_my_figs)

    if plot_time_series:
        lat = ds['lat'][:]
        lon = ds['lon'][:]
        index_lat = closestNode(lat_point, lat)  # only used in timeseries
        index_lon = closestNode(lon_point, lon)  # only used in timeseries
        sm_q = ds['SM-SMAP-L-DESC_V4.0_100'][0, index_lon, index_lat]
        if (bool(sm_q)):
            date_time_obj = datetime.datetime.strptime(ds.datetime, '%Y-%m-%d %H:%M:%S')
            df.loc[i] = [date_time_obj, sm_q]  # put data in dataframe
        else:
            sm_q = np.nan

    i += 1

if plot_time_series:
    df = df.set_index('date')  # https://www.dataquest.io/blog/tutorial-time-series-analysis-with-pandas/

    my_fontsize = 14
    year_size = 365  # approx 5 years of daily data
    df['soil_moisture'].plot(marker='.', ms=8, alpha=1, linestyle='None',
                             figsize=(5 * (math.ceil(df.size / year_size)), 5), fontsize=my_fontsize, grid=True)
    # # todo: smoothing average line (plots, but wrong date, add to dataframe?)
    # sm_av = movingaverage(df['soil_moisture'], 50)  # testing
    # df_tmp = pd.DataFrame({'sm_av': sm_av})
    # df.append(df_tmp)
    # df['sm_av'].plot('r')
    # plt.plot(sm_av,'r')
    plt.title('SMAP timeseries for queried coordinate', fontsize=my_fontsize)
    plt.xlabel('', fontsize=my_fontsize)
    plt.ylabel('Soil moisture (m$^3$/ m$^3$)', fontsize=my_fontsize)
    plt.tight_layout()

    if save_my_figs:
        plt.savefig(r'files/outputs/ts_sm_' + str(lon_point) + '_' + str(lat_point) + '.png', dpi=300)
    else:
        plt.show()

elapsed = time.time() - t
print(r'run time: ' + str(round(elapsed) / 60) + r' minutes')
