from libs.modules.my_methods import read_nz_reaches, read_topnet_soilh2o, compare_dataframes_smap_topnet, read_field_obs_soilh2o, compare_dataframes_obs_topnet
from libs.modules.utils import linear_regression_r2
import time
import os
import pandas as pd

t = time.time()

# step 1: read river reaches and their coordinates as a geopandas frame (output epsg: 2193)
gdf_path = os.path.join(os.getcwd(), r'files\dataframes')
gdf_file = 'nz_reaches_gdf'
roi_shape_fn = r'e:\\shapes\\Northland_NZTM.shp'

if not os.path.exists(os.path.join(gdf_path, gdf_file)):
    print('Building river reaches dataframe..')
    shape_fn = r'i:\\GroundWater\\Research\\NIWA_NationalHydrologyProgram\\Data\\SoilMoistureVanderSat\\TopnetFiles' \
                   r'\\GIS_DN2_3_Lake_strahler3\\rec2_3_order3_lakes_sites_watershed.shp'
    gdf_reaches = read_nz_reaches(shape_fn, roi_shape_fn)
else:
    print('Reading river reaches dataframe from ' + os.path.join(gdf_path, gdf_file))
    gdf_reaches = pd.read_pickle(os.path.join(gdf_path, gdf_file))

# step 2: read soil moisture (soilh2o) from a list of river catchments (or one catchment also works)
df_name_soilh2o = 'topnet_df'
df_path = os.path.join(os.getcwd(), r'files\dataframes')
df_name = os.path.join(df_path, df_name_soilh2o)
existing_pickle = True
input_rchids = None
if existing_pickle:
    print('Reading topnet soilh2o from dataframe object ' + os.path.join(df_path, df_name))
    df_topnet = pd.read_pickle(df_name)
else:
    rchids = list(gdf_reaches.index)
    input_rchids = rchids[:]

    my_path = r'i:\GroundWater\Research\NIWA_NationalHydrologyProgram\Data\SoilMoistureVanderSat\TopnetFiles'
    my_file = r'streamq_daily_average_2016060100_2021053100_utc_topnet_01000000_strahler1-SoilM-NM.nc'
    nc_fn = os.path.join(my_path, my_file)
    print('reading topnet soilh2o...')
    df_topnet = read_topnet_soilh2o(input_rchids, nc_fn) # also possible to plot and save, see function input

    if not os.path.exists(df_path):
        os.mkdir(df_path)

    df_topnet.to_pickle(df_name)

# step 3: read SMAP data within river reach polygon and compile for each rchid.
print('reading pickled SMAP soil moisture averaged per topnet catchment ...')
df_path = gdf_path
df_file = 'smap_per_reach_df0_1000'
df_a = pd.read_pickle(os.path.join(df_path, df_file))
df_file = 'smap_per_reach_df1000_2000'
df_b = pd.read_pickle(os.path.join(df_path, df_file))
df_file = 'smap_per_reach_df2000_2905'
df_c = pd.read_pickle(os.path.join(df_path, df_file))
df_smap = pd.concat([df_a, df_b, df_c], axis=1)
del df_a, df_b, df_c

df_smap = df_smap.tz_localize(None) # time stamp is now the same format as soilh2o_df

# # step 4: compare dataframes of smap to topnet
gdf_reaches_with_r2 = compare_dataframes_smap_topnet(df_smap, df_topnet, gdf_path, gdf_file, my_shape=roi_shape_fn,
                                                     plot_time_series=False)

# step 5: read field observations. Store as gdf
data_path = r'i:\GroundWater\Research\NIWA_NationalHydrologyProgram\Data\SoilMoistureVanderSat\SoilMoistureObservations'
data_fn = 'NZWaM_SM_DN3_2016-2021_20220412.nc'

gdf_obs, df_obs = read_field_obs_soilh2o(data_path, data_fn, roi_shape_fn)

# step 6: compare dataframes of topnet to field_obs
gdf_obs_with_r2s_topnet = compare_dataframes_obs_topnet(gdf_obs, df_obs, gdf_reaches, df_topnet,
                                                 plot_time_series=False, save_new_gdf=True,
                                                 plot_correlation_maps=False, my_shape=roi_shape_fn)

# step 7: read smap data at field observations
from libs.modules.my_methods import read_smap_at_obs

smap_at_obs_fn = os.path.join(os.getcwd(), r'files\dataframes\smap_at_obs_df')
if not os.path.exists(smap_at_obs_fn):
    df_smap_at_obs = read_smap_at_obs(gdf_obs)
    df_smap_at_obs.to_pickle(smap_at_obs_fn)
else:
    print('reading pickled SMAP data at field observation...')
    df_smap_at_obs = pd.read_pickle(smap_at_obs_fn)

df_smap_at_obs = df_smap_at_obs.tz_localize(None)


# step 8: compare dataframes of smap to field_obs
# todo: check why values are the same as topnet r2s???
print("todo: check why values are the same as topnet r2s???")
from libs.modules.my_methods import compare_smap_at_obs
gdf_obs_with_r2s_smap = compare_smap_at_obs(gdf_obs, df_obs, df_smap_at_obs, plot_time_series=False,
                                            plot_correlation_maps=False, my_shape=roi_shape_fn, save_new_gdf=True)

elapsed = time.time() - t
print(r'run time: ' + str(round(elapsed) / 60) + r' minutes')