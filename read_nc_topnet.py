# -*- coding: utf-8 -*-
"""
Read and plot Topnet data as provided in daily netcdfs by VanderSat

Developer: Rogier Westerhoff, GNS Science.
Topnet data produced by: NIWA. New Zealand

v001 - 2022-05-13: test for reading one file

todo: put in function; save as pickled df; choose start and end date
"""
# PyCharm: Press Shift+F10 to run script



import os
import netCDF4 as nc
import glob
import matplotlib.pyplot as plt
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
start_date = datetime.datetime(2016, 6, 1)
end_date = datetime.datetime(2016, 6, 30)  # year, month, day

# ++++++++ MY FUNCTIONS +++++++

from libs.modules.utils import indexContainingSubstring, closestNode, movingaverage
from libs.modules.my_methods import convert_nc_time
# +++++++ END OF FUNCTIONS +++++

work_dir = os.getcwd()
# EXAMPLE: my_path = os.path.join(work_dir, r'files\dataframe\')

t = time.time()

my_path = r'i:\GroundWater\Research\NIWA_NationalHydrologyProgram\Data\SoilMoistureVanderSat\TopnetFiles'
my_file = r'streamq_daily_average_2016060100_2021053100_utc_topnet_01000000_strahler1-SoilM-NM.nc'
# print(glob.glob(r"I:\GroundWater\Research\NIWA_NationalHydrologyProgram\Data\SoilMoistureVanderSat\NorthlandOutput\SMAP_LN\*.nc"))
fn = os.path.join(my_path, my_file)

ds = nc.Dataset(fn)
# print(ds)
# print(ds.__dict__)

# # to check values of smcsatn of all reaches:
# plt.hist(ds['smcsatn'][:], bins=50)
# plt.gca().set(title='smcsat (m/m$^2$)', ylabel='Frequency');
# plt.show()

i_reach = 8500 # nrch(29651) todo: find catchment closest to coordinate. Geopandas?
i_ens = 0 # not an ensemble, just one dataset
reach_id = np.ma.array(ds['rchid'][i_reach]) # int32 rchid(nrch)
soil_h2o = np.ma.array(ds['soilh2o'][:, i_reach, i_ens]) # float32 soilh2o(time, nrch, nens)
sm_saturated = np.ma.array(ds['smcsatn'][i_reach]) # float32 smcsatn(nrch)
sm_field_capacity = np.ma.array(ds['smfield'][i_reach]) # float32 smfield(nrch)
# print(sm)
sm_perc_of_smcsatn = 100*soil_h2o/sm_saturated #
column_names = ["soil_moisture"]
# df = pd.DataFrame(time, sm, columns=column_names)

# converting netcdf time
py_times = convert_nc_time(ds,'time') # float64 time(time)
# df = pd.DataFrame(index= py_times, data= sm_perc_of_smcsatn, columns= column_names) # if checking for sm_ratio_smcsatn
df = pd.DataFrame(index= py_times, data= soil_h2o, columns= column_names)

if plot_time_series:

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
    plt.title(r'TopNet timeseries for reach id ' + str(reach_id), fontsize=my_fontsize)
    plt.xlabel('', fontsize=my_fontsize)
    plt.ylabel('Topnet SM (m$^3$/m$^3$)', fontsize=my_fontsize)
    plt.tight_layout()

    if save_my_figs:
        plt.savefig(r'files/outputs/ts_sm_Topnet_reach_id_' + str(reach_id) + '.png', dpi=300)
    else:
        plt.show()

elapsed = time.time() - t
print(r'run time: ' + str(round(elapsed) / 60) + r' minutes')
