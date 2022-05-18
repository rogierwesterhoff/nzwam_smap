'''
imports geometries of TopNet reaches

author: Rogier Westerhoff, GNS Science, New Zealand
v001 - 20220517 - testing geopandas
'''
# run in PyCharm: Ctrl + Shift +F10

import geopandas
import time
import matplotlib.pyplot as plt
import os

t = time.time()

# +++++ INPUT (used for function later)
lat_point = -35.5
lon_point = 174.0
shape_fn = r'i:\\GroundWater\\Research\\NIWA_NationalHydrologyProgram\\Data\\SoilMoistureVanderSat\\TopnetFiles' \
               r'\\GIS_DN2_3_Lake_strahler3\\rec2_3_order3_lakes_sites_watershed.shp'
roi_shape_fn = r'e:\\shapes\\Northland_NZTM.shp'
plot_maps = True
save_to_pickle = False
# +++++ END INPUT

s = geopandas.read_file(shape_fn)
roi_shape = geopandas.read_file(roi_shape_fn)
roi_polygon = roi_shape.iloc[0].geometry # strip down to the actual polygon for use in .within()
# print(northland_polygon)

s = s.set_index("nzreach")

print(r'crs original file: ', str(s.crs))
# s = s.set_crs(4326, allow_override=True) # checked: not necessary, data already in 4326
s = s.to_crs(2193) # New Zealand Transverse Mercator (NZTM)
print(r'crs projected: ', str(s.crs))

# Only keep values for roi polygon
# filtering through polygon: https://stackoverflow.com/questions/46207530/filtering-pandas-dataframe-with-multiple-boolean-columns
s = s.loc[s.within(roi_polygon)]
# print('s:')
# print(s)
# print('s2:')
# print(s2)

# s3 = s.loc[s2]
# print('s3:')
# print(s3)

# s = s.within(northland_polygon)

s['area'] = s.area
s['area_ha'] = s.area/1e+4
# gdf.plot("area", legend=True)

# gdf['boundary'] = gdf.boundary
s['centroid'] = s.centroid
print(s['centroid'].head())

if plot_maps:
    # plot histogram of areas
    ax = s.plot(column='area', kind='hist', bins=50)
    plt.xlabel('Area (m$^2$)', fontsize=14)
    # plt.xlim(left=0, right=0.4e4)
    # plt.show()
    plt.savefig(r'files/outputs/reachnz_areas_hist.png', dpi=300)
    # # plots all centroids todo
    # s[:].plot('area', legend=True)
    # # plt.show()
    # plt.savefig(r'files/outputs/reachnz_areas_test.png', dpi=300)

work_dir = os.getcwd()
df_path = os.path.join(work_dir, r'files\dataframes')
if not os.path.exists(df_path):
    os.mkdir(df_path)

if save_to_pickle:
    df_output_fn = os.path.join(df_path, 'nz_reaches_df')
    s.to_pickle(df_output_fn)

elapsed = time.time() - t
print(r'run time: ' + str(round(elapsed) / 60) + r' minutes')