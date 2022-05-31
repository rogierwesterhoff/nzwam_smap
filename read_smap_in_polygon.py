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
import maya
import geopandas as gpd

shapely.speedups.enable()

# ++++++ FUNCTIONS
from libs.modules.utils import indexContainingSubstring, closestNode

# +++++++ END OF FUNCTIONS +++++

t = time.time()

# +++++ INPUT VARS
save_to_pickle = True

start_date = datetime.datetime(2016, 6, 1)
end_date = datetime.datetime(2021, 5, 31)  # year, month, day
data_path = r'i:GroundWater\Research\NIWA_NationalHydrologyProgram\Data\SoilMoistureVanderSat\SmapData\NorthlandNcFiles'

gdf_path = os.path.join(os.getcwd(), r'files\dataframes')
gdf_file = 'nz_reaches_gdf'

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

# identify river reaches
gdf_reaches = pd.read_pickle(os.path.join(gdf_path, gdf_file))
# print(r'crs original file: ', str(gdf_reaches.crs))
gdf_reaches = gdf_reaches.to_crs(4326)
# print(r'crs projected: ', str(gdf_reaches.crs))
rchids = list(gdf_reaches.index)

# start loop
# todo: check if pre-allocation all columns helps with speed/memory
for reach_idx in range(len(rchids)):
# for reach_idx in range(3):
    rchid = rchids[reach_idx]
    # define an empty pandas dataframe for dates and soil moisture values for the rchid

    column_names = ["Time", str(rchid)]
    df_in_loop = pd.DataFrame(columns=column_names)

    reach_polygon = gdf_reaches.iloc[reach_idx].geometry
    reach_bounds = reach_polygon.bounds

    elapsed = time.time() - t
    print(r"reach number: " + str(reach_idx+1) + " of " + str(len(rchids)))
    print(f"time elapsed: {round(elapsed) / 60:.3f} minutes")

    i = 0
    lat = []
    lon = []
    for ifile in range(index1, index2 + 1):

        fn = file_list[ifile]
        ds = nc.Dataset(fn)
        date_label = ds.datetime[0:10]

        if ifile == index1:
            print(r'header first file: ' + date_label)

        if ifile == index2:
            print(r'header last file: ' + ds.datetime[0:10])

        # only do this when SMAP filesize is different (hopefully only once since it takes long..)
        if len(ds['lat'][:]) != len(lat) or len(ds['lon'][:]) != len(lon):
            # print('Reading axes of smap nc in ifile '+str(ifile))
            lon = ds['lon'][:]
            lat = ds['lat'][:]
            # find bounding box of reach and only load those (saves a heap of time)
            index_lon_min = closestNode(reach_bounds[0], lon)  # only used in timeseries
            index_lon_max = closestNode(reach_bounds[2], lon)  # only used in timeseries
            index_lat_min = closestNode(reach_bounds[3], lat)  # only used in timeseries
            index_lat_max = closestNode(reach_bounds[1], lat)  # only used in timeseries
            # meshgrid and flatten in order to find points in polygon
            lon_grid, lat_grid = np.meshgrid(lon[index_lon_min:index_lon_max+1], lat[index_lat_min:index_lat_max+1])
            df_coords = pd.DataFrame(lat_grid.compressed(), columns=['Latitude'])
            df_coords['Longitude'] = lon_grid.compressed()
            points_gdf = gpd.GeoDataFrame(df_coords, geometry=gpd.points_from_xy(df_coords.Longitude, df_coords.Latitude))
            # elapsed = time.time() - t
            # print(f"time elapsed: {round(elapsed) / 60:.3f} minutes")

        sm = ds['SM-SMAP-L-DESC_V4.0_100'][0, index_lon_min:index_lon_max+1, index_lat_min:index_lat_max+1]
        # sm = sm.filled(np.nan).flatten()
        gdf_tmp = points_gdf
        gdf_tmp['soil moisture'] = sm.filled(np.nan).flatten()
        gdf_tmp = gdf_tmp.loc[points_gdf.within(reach_polygon)]
        sm_reach = gdf_tmp['soil moisture'].mean()
        # print(f"sm_reach: {sm_reach:.3f} m3/m3")
        # date_time_obj = datetime.datetime.strptime(ds.datetime, '%Y-%m-%d %H:%M:%S')

        date_time_obj = maya.parse(ds.datetime).datetime()

        # if not bool(sm_reach):
        #     sm_reach = np.nan

        df_in_loop.loc[i] = [date_time_obj, sm_reach]  # put data in dataframe

        # print('gdf filtered for reach...')
        # elapsed = time.time() - t
        # print(f"time elapsed: {round(elapsed) / 60:.3f} minutes")

        i += 1

    # the below line adds the time column for every reach. That might seem inefficient,
    # but it does open up the possibility for timeseries with different time intervals.
    df_in_loop.set_index('Time', inplace=True)
    if reach_idx == 0:
        df_whole = df_in_loop
    else:
        df_whole = pd.concat([df_whole,df_in_loop], axis = 1)

# df_whole['Time'] = pd.to_datetime(df_whole['Time'], dayfirst=True) # todo: check if this is necessary. I might have already been done by maya
# # set the index to timestamp
# df_whole.set_index('Time', inplace=True)

if save_to_pickle:
    work_dir = os.getcwd()
    df_path = os.path.join(work_dir, r'files\dataframes')
    if not os.path.exists(df_path):
        os.mkdir(df_path)

    df_output_fn = os.path.join(df_path, 'smap_per_reach_df')
    df_whole.to_pickle(df_output_fn)
    print(r'GeoDataframe saved in files/dataframes.')

# # --- store as pickle ----
# work_dir = os.getcwd()  # might be needed if stored as a dataframe and grabbed later
# df_path = os.path.join(work_dir, r'files\dataframes')
# if not os.path.exists(df_path):
#     os.mkdir(df_path)
# df_filename = os.path.join(df_path, 'smap_df')
# df.to_pickle(df_filename)
# print(r'Dataframe pickled in: ' + df_filename)
elapsed = time.time() - t
print(f"total run time: {round(elapsed) / 60:.3f} minutes")
