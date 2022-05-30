from libs.modules.my_methods import readNcSmapToDf, read_nz_reaches, read_topnet_soilh2o
import time
import os
import pandas as pd

t = time.time()

# step 1: read river reaches and their coordinates as a geopandas frame (output epsg: 2193)
gdf_path = os.path.join(os.getcwd(), r'files\dataframes')
gdf_file = 'nz_reaches_gdf'

if not os.path.exists(os.path.join(gdf_path, gdf_file)):
    shape_fn = r'i:\\GroundWater\\Research\\NIWA_NationalHydrologyProgram\\Data\\SoilMoistureVanderSat\\TopnetFiles' \
                   r'\\GIS_DN2_3_Lake_strahler3\\rec2_3_order3_lakes_sites_watershed.shp'
    roi_shape_fn = r'e:\\shapes\\Northland_NZTM.shp'
    gdf_reaches = read_nz_reaches(shape_fn, roi_shape_fn)
else:
    print('Reading dataframe object from ' + os.path.join(gdf_path, gdf_file))
    gdf_reaches = pd.read_pickle(os.path.join(gdf_path, gdf_file))

# step 2: read a specific river catchment
rchids = list(gdf_reaches.index)
rchid = rchids[0]

# todo: a loop over the rchids? maybe, but decide only at the very end
my_path = r'i:\GroundWater\Research\NIWA_NationalHydrologyProgram\Data\SoilMoistureVanderSat\TopnetFiles'
my_file = r'streamq_daily_average_2016060100_2021053100_utc_topnet_01000000_strahler1-SoilM-NM.nc'
nc_fn = os.path.join(my_path, my_file)
soilh2o_df = read_topnet_soilh2o(rchid, nc_fn) # also possible to plot and save, see function input

# step 3: read SMAP data within river reach polygon and compile for each rchid. Store as gdf
df_path = gdf_path
df_file = 'smap_per_reach_df'
smap_df = pd.read_pickle(os.path.join(df_path, df_file))

# step 4: merge dataframes

# step 5: read field observations and look for closest SMAP pixel(s?). Store as gdf

# step 6: now go do some data science!


# start_date = datetime.datetime(2016, 6, 1)
# end_date = datetime.datetime(2016, 6, 30)  # year, month, day
# data_path = r'i:GroundWater\Research\NIWA_NationalHydrologyProgram\Data\SoilMoistureVanderSat\SmapData\NorthlandNcFiles'
# readNcSmapToDf(lat_point, lon_point, start_date, end_date, data_path)

elapsed = time.time() - t
print(r'run time: ' + str(round(elapsed) / 60) + r' minutes')