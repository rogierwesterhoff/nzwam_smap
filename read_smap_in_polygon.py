"""
 reads SMAP data from a netcdf file and stores timeseries within a polygon as a pandas dataframe
 example to run:

 """

import os
import netCDF4 as nc
import glob
import numpy as np
import datetime
import pandas as pd
import time
import shapely.speedups
shapely.speedups.enable()
from shapely.geometry import Polygon
import geopandas as gpd
from itertools import compress
from matplotlib.path import Path

t = time.time()

# +++++ INPUT VARS
start_date = datetime.datetime(2016, 6, 1)
end_date = datetime.datetime(2016, 6, 30)  # year, month, day
data_path = r'i:GroundWater\Research\NIWA_NationalHydrologyProgram\Data\SoilMoistureVanderSat\SmapData\NorthlandNcFiles'

lat_point = -35.5 # todo: remove once within polygon is sorted
lon_point = 174.0

gdf_path = os.path.join(os.getcwd(), r'files\dataframes')
gdf_file = 'nz_reaches_gdf'
gdf_reaches = pd.read_pickle(os.path.join(gdf_path, gdf_file))

# step 2: read a specific river catchment
reach_idx = 0

rchids = list(gdf_reaches.index)
rchid = rchids[reach_idx]

print(r'crs original file: ', str(gdf_reaches.crs))
gdf_reaches = gdf_reaches.to_crs(4326)
print(r'crs projected: ', str(gdf_reaches.crs))
reach_polygon = gdf_reaches.iloc[reach_idx].geometry
print(reach_polygon)

# ++++++ FUNCTIONS
from libs.modules.utils import indexContainingSubstring, closestNode, inpolygon

# +++++++ END OF FUNCTIONS +++++

# print(glob.glob(r"I:\GroundWater\Research\NIWA_NationalHydrologyProgram\Data\SoilMoistureVanderSat\NorthlandOutput\SMAP_LN\*.nc"))
file_list = sorted(glob.glob(os.path.join(data_path, r'SM-SMAP*.nc')))

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
column_names = ["Time", "soil_moisture"]
df = pd.DataFrame(columns=column_names)

i = 0
# start loop
# for ifile in range(index1, index2 + 1):
for ifile in range(index1, index1 + 1):

    fn = file_list[ifile]
    ds = nc.Dataset(fn)
    date_label = ds.datetime[0:10]

    if ifile == index1:
        print(r'header first file: ' + date_label)

    if ifile == index2:
        print(r'header last file: ' + ds.datetime[0:10])

    # todo: only do this when filesize is different (hopefully only once then..)
    lat = ds['lat'][:]
    lon = ds['lon'][:]

    lon_grid, lat_grid = np.meshgrid(lon, lat)
    # lats = np.transpose(lat_grid.compressed()) #.flatten()
    # lons = np.transpose(lon_grid.compressed()) #.flatten()

    lats = lat_grid.compressed()  # .flatten()
    lons = lon_grid.compressed()  # .flatten()

    df_coords = pd.DataFrame(lats, columns = ['Latitude'])
    df_coords['Longitude'] = lons
    points_gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df_coords.Longitude, df_coords.Latitude))
    print('gdf of all points in mesh defined...')
    elapsed = time.time() - t
    print(r'time elapsed: ' + str(round(elapsed) / 60) + r' minutes')

    points_gdf = points_gdf.loc[points_gdf.within(reach_polygon)]
    print('gdf filtered for reach...')
    elapsed = time.time() - t
    print(r'time elapsed: ' + str(round(elapsed) / 60) + r' minutes')

    # in_points2 = points.loc[in_points]
    # print(reach_polygon[:,0])

    # todo: below is still from the old file. Read data in for each coordinate in \
    #  points_gdf and then take the mean + std
    index_lat = closestNode(lat_point, lat)  # only used in timeseries
    index_lon = closestNode(lon_point, lon)  # only used in timeseries
    sm = ds['SM-SMAP-L-DESC_V4.0_100'][0, index_lon, index_lat]
    sm = np.ma.array(sm)  # ma because of the masked array
    if bool(sm):  # if not empty
        date_time_obj = datetime.datetime.strptime(ds.datetime, '%Y-%m-%d %H:%M:%S')
        df.loc[i] = [date_time_obj, sm]  # put data in dataframe
    else:
        sm = np.nan

    i += 1

df['Time'] = pd.to_datetime(df['Time'], dayfirst=True)
# set the index to timestamp
df.set_index('Time', inplace=True)

# # --- store as pickle ----
# work_dir = os.getcwd()  # might be needed if stored as a dataframe and grabbed later
# df_path = os.path.join(work_dir, r'files\dataframes')
# if not os.path.exists(df_path):
#     os.mkdir(df_path)
# df_filename = os.path.join(df_path, 'smap_df')
# df.to_pickle(df_filename)
# print(r'Dataframe pickled in: ' + df_filename)
elapsed = time.time() - t
print(r'total rune time: ' + str(round(elapsed) / 60) + r' minutes')
