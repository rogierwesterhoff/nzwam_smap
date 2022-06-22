"""
 reads field observations from netCDF file

 """

import os
import netCDF4 as nc
import glob

import numpy
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import time
import maya
import geopandas as gpd

data_path = r'i:\GroundWater\Research\NIWA_NationalHydrologyProgram\Data\SoilMoistureVanderSat\SoilMoistureObservations'
data_file = 'NZWaM_SM_DN3_2016-2021_20220412.nc'

fn = os.path.join(data_path,data_file)
ds = nc.Dataset(fn)
print(ds)
# read: soilh2o(time, station), station_rchid(station), latitude(station), longitude(station)
# todo: write a pandas df for time (index), station_rchid (columns)
# soil_h2o = np.ma.array(ds['soilh2o'][:, reach_idx, ens_idx])  # float32 soilh2o(time, nrch, nens)
soil_h2o = np.ma.array(ds['soilh2o'][1, :]) # float64 soilh2o(time, station) in mm [0 - 1000]
sample_times = np.ma.array(ds['time'][:]) # int64 time(time). units: hours since 2015-12-31 12:00:00+00:00, calendar: proleptic_gregorian
lons = np.ma.array(ds['longitude'][:]) # float64 longitude(station)
lats = np.ma.array(ds['latitude'][:]) # float64 latitude(station)

# todo: write pytimes based on units and calendar

plt.plot(sample_times)
plt.show()

# todo: write a geopandas to store information on all soil moisture stations
print(type(soil_h2o))