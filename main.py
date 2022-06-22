from libs.modules.my_methods import readNcSmapToDf, read_nz_reaches, read_topnet_soilh2o
from libs.modules.utils import linear_regression_r2
import time
import os
import pandas as pd

t = time.time()

# step 1: read river reaches and their coordinates as a geopandas frame (output epsg: 2193)
gdf_path = os.path.join(os.getcwd(), r'files\dataframes')
gdf_file = 'nz_reaches_gdf'

if not os.path.exists(os.path.join(gdf_path, gdf_file)):
    print('Building river reaches dataframe..')
    shape_fn = r'i:\\GroundWater\\Research\\NIWA_NationalHydrologyProgram\\Data\\SoilMoistureVanderSat\\TopnetFiles' \
                   r'\\GIS_DN2_3_Lake_strahler3\\rec2_3_order3_lakes_sites_watershed.shp'
    roi_shape_fn = r'e:\\shapes\\Northland_NZTM.shp'
    gdf_reaches = read_nz_reaches(shape_fn, roi_shape_fn)
else:
    print('Reading river reaches dataframe from ' + os.path.join(gdf_path, gdf_file))
    gdf_reaches = pd.read_pickle(os.path.join(gdf_path, gdf_file))

# step 2: read soil moisture (soilh2o) from a list of river catchments (or one catchment also works)
df_name_soilh2o = 'soilh2o_df'
df_path = os.path.join(os.getcwd(), r'files\dataframes')
df_name = os.path.join(df_path, df_name_soilh2o)
existing_pickle = True
if existing_pickle:
    print('Reading topnet soilh2o from dataframe object ' + df_path)
    topnet_df = pd.read_pickle(df_name)
else:
    rchids = list(gdf_reaches.index)
    input_rchids = rchids[:]

    my_path = r'i:\GroundWater\Research\NIWA_NationalHydrologyProgram\Data\SoilMoistureVanderSat\TopnetFiles'
    my_file = r'streamq_daily_average_2016060100_2021053100_utc_topnet_01000000_strahler1-SoilM-NM.nc'
    nc_fn = os.path.join(my_path, my_file)
    print('reading topnet soilh2o...')
    topnet_df = read_topnet_soilh2o(input_rchids, nc_fn) # also possible to plot and save, see function input

    if not os.path.exists(df_path):
        os.mkdir(df_path)

    topnet_df.to_pickle(df_name)

# step 3: read SMAP data within river reach polygon and compile for each rchid.
print('reading SMAP soil moisture (files were pre-processed earlier) ...')
df_path = gdf_path
df_file = 'smap_per_reach_df0_1000'
df_a = pd.read_pickle(os.path.join(df_path, df_file))
df_file = 'smap_per_reach_df1000_2000'
df_b = pd.read_pickle(os.path.join(df_path, df_file))
df_file = 'smap_per_reach_df2000_2905'
df_c = pd.read_pickle(os.path.join(df_path, df_file))
smap_df = pd.concat([df_a, df_b, df_c], axis=1)
del df_a, df_b, df_c

smap_df = smap_df.tz_localize(None) # time stamp is now the same format as soilh2o_df

# step 4: compare dataframes
# todo: put in function when done testing
# def compare_dataframes(smap_df, topnet_df, gdf_path, gdf_file):
import matplotlib.pyplot as plt
import numpy as np
import math
plot_time_series = False
save_new_gdf = True
plot_correlation_maps = True

gdf_reaches = pd.read_pickle(os.path.join(gdf_path, gdf_file))
rchids = list(gdf_reaches.index)
input_rchids = rchids[:]

my_r2s = [np.nan] * len(rchids)

if not input_rchids[0] == rchids[0]:
    raise Exception("input rchid has to start with 0 to follow the first input of gdf_reaches")

if isinstance(input_rchids, int):
    input_rchids = [input_rchids]

# round dates to day so they can be compared
smap_df.index = smap_df.index.floor('D')
topnet_df.index = topnet_df.index.floor('D')

for i in range(len(input_rchids)):
    rchid = input_rchids[i]
    # print('rchid = ' + str(rchid))
    smap_col_df = smap_df[str(rchid)]
    topnet_col_df = topnet_df[str(rchid)]
    if np.sum(smap_col_df.count()) > 0 and np.sum(topnet_col_df.count()) > 0:
        smap_col_df = smap_col_df.fillna('NaN').astype('float')  # convert NaT to NaN and cast values to float
        topnet_col_df = topnet_col_df.fillna('NaN').astype('float')  # convert NaT to NaN and cast values to float

        frames = [smap_col_df, topnet_col_df]
        joint_df = pd.concat(frames, axis=1)
        joint_df.dropna(inplace=True)
        joint_df.columns = [r'smap_'+str(rchid), r'topnet_'+str(rchid)]
        # calculate R2 and add to gdfreaches as a column
        # https://www.statology.org/r-squared-in-python/
        # R2 values of:
        # - smap to topnet (done)
        # - use that to interpolate missing smap
        # - smap to field observations(with and without interpolate)
        # - topnet to field observations
        # - smap and topnet to field observations(with and without interpolate)

        # calculate R-squared of regression model
        r_squared = linear_regression_r2(joint_df[r'smap_'+str(rchid)], joint_df[r'topnet_'+str(rchid)])
        # view R-squared value
        # print(r_squared)
        my_r2s[i] = r_squared

        if plot_time_series:
            print('plot rchid = ' + str(rchid))
            saveFigName = r'rchid_' + str(rchid) + '_topnet_smap'
            my_fontsize = 14
            year_size = 365  # approx 5 years of daily data

            ax = smap_col_df.plot(marker='.', ms=5, alpha=1, linestyle='None',
                                     figsize=(5 * (math.ceil(smap_col_df.size / year_size)), 5),
                                  fontsize=my_fontsize, grid=True, label='smap')
            topnet_col_df.plot(ax=ax, label='topnet')
            plt.legend(loc='best')
            plt.title(r'soil moisture in reach id ' + str(rchid), fontsize=my_fontsize)
            plt.xlabel('', fontsize=my_fontsize)
            plt.ylabel('SM (m$^3$/m$^3$)', fontsize=my_fontsize)
            plt.tight_layout()
            plt.tight_layout()
            fig_path = os.path.join(os.getcwd(), r'files\outputs')
            if not os.path.exists(fig_path):
                os.mkdir(fig_path)
            saveFigName = os.path.join(fig_path, saveFigName)
            plt.savefig(saveFigName + '.png', dpi=300)
            # plt.savefig(saveFigName + '.eps', dpi=300)
            plt.close()
            # plt.show()

if save_new_gdf:
    gdf_reaches['r2_smap_topnet']=my_r2s
    gdf_reaches.to_pickle(os.path.join(gdf_path, gdf_file+'_r2'))

if plot_correlation_maps:
    gdf = gdf_reaches.set_geometry("centroid")
    gdf.dropna().plot("r2_smap_topnet", legend=True, markersize=5)
    # plt.show()
    saveFigName = 'r2_smap_topnet'
    fig_path = os.path.join(os.getcwd(), r'files\outputs')
    if not os.path.exists(fig_path):
        os.mkdir(fig_path)
    plt.savefig(os.path.join(fig_path, saveFigName) + '.png', dpi=300)
    plt.close()
    
# step 5: read field observations and look for closest SMAP pixel(s?). Store as gdf

# step 6: now go do some data science!


# start_date = datetime.datetime(2016, 6, 1)
# end_date = datetime.datetime(2016, 6, 30)  # year, month, day
# data_path = r'i:GroundWater\Research\NIWA_NationalHydrologyProgram\Data\SoilMoistureVanderSat\SmapData\NorthlandNcFiles'
# readNcSmapToDf(lat_point, lon_point, start_date, end_date, data_path)

elapsed = time.time() - t
print(r'run time: ' + str(round(elapsed) / 60) + r' minutes')