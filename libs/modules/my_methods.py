import numpy as np
import matplotlib.pyplot as plt
import os
import netCDF4 as nc
import glob
import datetime
import time
import pandas as pd
import geopandas
import math


def plotSmMaps(data, my_extent, labelstr, defaultScale, save_my_fig):
    # minVal = np.percentile(data.compressed(),5) #min_value = np.min(sm)
    # maxVal = np.percentile(data.compressed(),95) #max_value = np.max(sm)

    if defaultScale:  # UGLY!! TRY args and kwargs (but having trouble passing kwargs as variable names)
        minVal = 0
        maxVal = 0.5
    else:
        maxVal = np.max(data)
        minVal = np.min(data)

    lon_min = my_extent[0]
    lon_max = my_extent[1]
    lat_min = my_extent[2]
    lat_max = my_extent[3]

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.set_aspect('equal')
    smplot = plt.imshow(data, cmap=plt.cm.terrain_r, interpolation='none',
                        #       extent=[lon_min,lon_max,lat_min,lat_max],vmin=0.15,vmax=0.45)
                        extent=my_extent, vmin=minVal, vmax=maxVal)
    plt.text(lon_max - 0.05, lat_max - 0.1, labelstr, ha='right')  # , bbox={'facecolor': 'white', 'pad': 10})

    cbar = fig.colorbar(smplot, ax=ax, anchor=(0, 0.3), shrink=0.8)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('m $^3$ m $^{-3}$', rotation=270, va='top')
    if save_my_fig:
        plt.savefig(r'files/outputs/sm_' + labelstr + '.png', dpi=300)
    else:
        plt.show()


def readNcSmapToDf(lat_point, lon_point, start_date, end_date, data_path):
    """
    reads SMAP data from a netcdf file and stores it as a pandas dataframe
    example to run:
    lat_point = -35.5
    lon_point = 174.0
    start_date = datetime.datetime(2016, 6, 1)
    end_date = datetime.datetime(2016, 6, 30)  # year, month, day
    data_path = myfolder

    readNcSmapToDf(lat_point, lon_point, start_date, end_date, data_path)
    """

    from libs.modules.utils import indexContainingSubstring, closestNode

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
    for ifile in range(index1, index2 + 1):

        fn = file_list[ifile]
        ds = nc.Dataset(fn)
        date_label = ds.datetime[0:10]

        if ifile == index1:
            print(r'header first file: ' + date_label)

        if ifile == index2:
            print(r'header last file: ' + ds.datetime[0:10])

        lat = ds['lat'][:]
        lon = ds['lon'][:]

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

    # --- store as pickle ----
    work_dir = os.getcwd()  # might be needed if stored as a dataframe and grabbed later
    df_path = os.path.join(work_dir, r'files\dataframes')
    if not os.path.exists(df_path):
        os.mkdir(df_path)

    df_filename = os.path.join(df_path, 'smap_df')
    df['Time'] = pd.to_datetime(df['Time'], dayfirst=True)
    # set the index to timestamp
    df.set_index('Time', inplace=True)
    df.to_pickle(df_filename)

    print(r'Dataframe pickled in: ' + df_filename)


def read_nz_reaches(shape_fn, roi_shape_fn, plot_maps=False, save_to_pickle=True):
    print(r'Creating gdf for reaches in roi...')
    s = geopandas.read_file(shape_fn)
    roi_shape = geopandas.read_file(roi_shape_fn)
    roi_polygon = roi_shape.iloc[0].geometry  # strip down to the actual polygon for use in .within()
    # print(northland_polygon)

    s = s.set_index("nzreach")

    print(r'crs original file: ', str(s.crs))
    # s = s.set_crs(4326, allow_override=True) # checked: not necessary, data already in 4326
    s = s.to_crs(2193)  # New Zealand Transverse Mercator (NZTM)
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
    s['area_ha'] = s.area / 1e4
    # gdf.plot("area", legend=True)

    # gdf['boundary'] = gdf.boundary
    s['centroid'] = s.centroid
    # print(s['centroid'].head())

    if plot_maps:
        my_fontsize = 12
        # plot histogram of areas
        ax = s.plot(column='area', kind='hist', bins=50, xlim=[0, 3e7], grid=True)
        ax.set_xlabel('Area (m$^2$)', fontsize=my_fontsize)
        ax.set_ylabel('# of catchments', fontsize=my_fontsize)
        # plt.xlabel('Area (m$^2$)', fontsize=my_fontsize)
        # plt.xlim(left=0, right=4e+7)
        # plt.show()
        plt.savefig(r'files/outputs/reach_roi_areas_hist.png', dpi=300)
        plt.close()

        # plots all centroids and geometry
        s = s.set_geometry(
            'centroid')  # set geometry to centroid for plotting (https://geopandas.org/en/stable/getting_started/introduction.html)
        ax = s['geometry'].plot()
        ax.set_xlabel('Easting (NZTM)', fontsize=my_fontsize)
        ax.set_ylabel('Northing (NZTM)', fontsize=my_fontsize)
        s['centroid'].plot(ax=ax, color='black', markersize=1)
        # s[:].plot('area', legend=True)
        # plt.show()
        plt.savefig(r'files/outputs/reach_roi_areas_test.png', dpi=300)

        s = s.set_geometry('geometry')  # set geometry back to the original geoseries
        print(r'Plots saved in files/outputs.')

    if save_to_pickle:
        work_dir = os.getcwd()
        df_path = os.path.join(work_dir, r'files\dataframes')
        if not os.path.exists(df_path):
            os.mkdir(df_path)

        gdf_output_fn = os.path.join(df_path, 'nz_reaches_gdf')
        s.to_pickle(gdf_output_fn)
        print(r'GeoDataframe saved in files/dataframes.')

    return s
    # elapsed = time.time() - t
    # print(r'run time: ' + str(round(elapsed) / 60) + r' minutes')


def read_topnet_soilh2o(reach_numbers, nc_fn, plot_time_series=False, save_my_figs=False):
    """
    Read and plot Topnet data as provided by NIWA

    Developer: Rogier Westerhoff, GNS Science.
    Topnet data produced by: NIWA. New Zealand

    v001 - 2022-05-13: test for reading one file
    v002 - embedded as function to reach in based on reach id number (rchid)
    v003 - 2022-06-21: changed input such that reach_numbers input can be one integer or list

    reach_numbers: reach ids as known in Topnet reaches file (NIWA)
    nc_fn: filename (netCDF) that contains the soil moisture timeseries
    plot_time_series: whether or not to plot timeseries (default = False)
    save_my_figs: whether or not to save the figure (default = False)
    """

    # ++++++++ IMPORT FUNCTIONS +++++++
    from libs.modules.utils import convert_nc_time

    # +++++++ END OF FUNCTIONS +++++

    work_dir = os.getcwd()

    ds = nc.Dataset(nc_fn)
    # print(ds)
    # print(ds.__dict__)
    rchids = np.ma.array(ds['rchid'][:])  #

    d = {}
    # start loop over reach numbers here
    if isinstance(reach_numbers, int):
        reach_numbers = [reach_numbers]

    for i in range(len(reach_numbers)):
        # print(r'i = ' + str(i))
        reach_number = int(reach_numbers[i])
        # print(r'reading soil moisture timeseries for reach: ' + str(reach_number))

        reach_idx = np.where(rchids == reach_number)

        try:
            if np.sum(reach_idx) == 0:
                print('Warning: Topnet rchid number ', str(reach_number), ' does not exist (i= ', str(i), ').')
            reach_idx = int(reach_idx[0])
            ens_idx = 0  # not an ensemble, just one dataset
        except:
            pass

        reach_id = np.ma.array(ds['rchid'][reach_idx])  # int32 rchid(nrch)
        soil_h2o = np.ma.array(ds['soilh2o'][:, reach_idx, ens_idx])  # float32 soilh2o(time, nrch, nens)
        sm_saturated = np.ma.array(ds['smcsatn'][reach_idx])  # float32 smcsatn(nrch)
        sm_field_capacity = np.ma.array(ds['smfield'][reach_idx])  # float32 smfield(nrch)
        sm_perc_of_smcsatn = 100 * soil_h2o / sm_saturated  #

        # print(f"mean soil moisture: {np.nanmean(soil_h2o):.3f} m3/m3")

        # converting netcdf time
        py_times = convert_nc_time(ds, 'time')  # float64 time(time)
        d[reach_number] = pd.DataFrame(
            {
                'Time': py_times,
                str(reach_number): soil_h2o,
            },
            # index = pytimes,
        )

        d[reach_number].set_index('Time', inplace=True)

    df = pd.concat(d.values(), axis=1)

    if plot_time_series:

        my_fontsize = 14
        year_size = 365  # approx 5 years of daily data
        df['soil_moisture'].plot(marker='.', ms=8, alpha=1, linestyle='None',
                                 figsize=(5 * (math.ceil(df.size / year_size)), 5), fontsize=my_fontsize, grid=True)
        plt.title(r'TopNet timeseries for reach id ' + str(reach_id), fontsize=my_fontsize)
        plt.xlabel('', fontsize=my_fontsize)
        plt.ylabel('Topnet SM (m$^3$/m$^3$)', fontsize=my_fontsize)
        plt.tight_layout()

        if save_my_figs:
            plt.savefig(r'files/outputs/ts_sm_Topnet_reach_id_' + str(reach_id) + '.png', dpi=300)
        else:
            plt.show()

    return df


def read_field_obs_soilh2o(data_path, data_fn, roi_shape_fn, plot_maps=False, save_to_pickle=True):
    import os
    import netCDF4 as nc
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import geopandas as gpd

    from libs.modules.utils import convert_nc_time

    # data_path = r'i:\GroundWater\Research\NIWA_NationalHydrologyProgram\Data\SoilMoistureVanderSat\SoilMoistureObservations'
    # data_fn = 'NZWaM_SM_DN3_2016-2021_20220412.nc'
    # roi_shape_fn = r'e:\\shapes\\Northland_NZTM.shp'
    # plot_maps = False
    # save_to_pickle = True

    fn = os.path.join(data_path, data_fn)
    ds = nc.Dataset(fn)
    # print(ds)

    # write a pandas df for time (index), station_rchid (columns) and soilh2o
    # soil_h2o = np.ma.array(ds['soilh2o'][:, reach_idx, ens_idx])  # float32 soilh2o(time, nrch, nens)
    station_id = np.ma.array(ds['station'][:])
    station_rchid = np.ma.array(ds['station_rchid'][:])
    soil_h2o = np.ma.array(ds['soilh2o'][:, :])  # float64 soilh2o(time, station) in mm [0 - 1000]
    sample_times = np.ma.array(
        ds['time'][:])  # int64 time(time). units: hours since 2015-12-31 12:00:00+00:00, calendar: proleptic_gregorian
    lons = np.ma.array(ds['longitude'][:])  # float64 longitude(station)
    lats = np.ma.array(ds['latitude'][:])  # float64 latitude(station)

    # write pytimes
    py_times = convert_nc_time(ds, 'time')

    df = pd.DataFrame(columns=station_rchid, data=soil_h2o / 1000, index=py_times)
    df = df.shift(periods=12, freq="H")  # shift 12 hours

    # write a geopandas to store information on all soil moisture stations
    df_tmp = pd.DataFrame(
        {'Station': station_id,
         'Station Rchid': station_rchid,
         'Latitude': lats,
         'Longitude': lons})

    gdf = gpd.GeoDataFrame(df_tmp, geometry=gpd.points_from_xy(df_tmp.Longitude, df_tmp.Latitude)) \
        .set_crs(4326, allow_override=True, inplace=True)
    # print(r'crs gdf: ', str(gdf.crs))

    # read in roi and define what obs are within polygon
    s = gpd.read_file(roi_shape_fn).to_crs(4326)  # print(r'crs roi shapefile: ', str(s.crs))
    gdf_in = gdf.within(s.iloc[0].geometry)  # strip s down to the polygon

    if plot_maps:  # plot map with field obs in shape
        my_fontsize = 12
        ax = s.plot(color='white', edgecolor='black')
        gdf[gdf_in].plot(ax=ax, color='red')
        ax.set_xlabel('Longitude', fontsize=my_fontsize)
        ax.set_ylabel('Latitude', fontsize=my_fontsize)
        plt.savefig(r'files/outputs/locations_field_obs.png', dpi=300)
        plt.close()

    if save_to_pickle:
        work_dir = os.getcwd()
        df_path = os.path.join(work_dir, r'files\dataframes')
        if not os.path.exists(df_path):
            os.mkdir(df_path)

        gdf_output_fn = os.path.join(df_path, 'field_obs_roi_gdf')
        gdf[gdf_in].to_pickle(gdf_output_fn)

        df_output_fn = os.path.join(df_path, 'field_obs_roi_df')
        df[station_rchid[gdf_in]].to_pickle(gdf_output_fn)

        gdf_output_fn = os.path.join(df_path, 'field_obs_nz_gdf')
        gdf.to_pickle(gdf_output_fn)

        df_output_fn = os.path.join(df_path, 'field_obs_nz_df')
        df.to_pickle(gdf_output_fn)

        # print(r'GeoDataframes saved in files/dataframes.')

    return gdf[gdf_in], df[station_rchid[gdf_in]]


def compare_dataframes_smap_topnet(smap_df, topnet_df, gdf_path, gdf_file, plot_time_series=False, save_new_gdf=True,
                                   plot_correlation_maps=True, my_shape=None):
    import matplotlib.pyplot as plt
    import numpy as np
    import math
    import geopandas as gpd
    from libs.modules.utils import linear_regression_r2

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
            joint_df.columns = [r'smap_' + str(rchid), r'topnet_' + str(rchid)]
            # calculate R2 and add to gdfreaches as a column
            # https://www.statology.org/r-squared-in-python/
            # R2 values of:
            # - smap to topnet (done)
            # - use that to interpolate missing smap
            # - smap to field observations(with and without interpolate)
            # - topnet to field observations
            # - smap and topnet to field observations(with and without interpolate)

            # calculate R-squared of regression model
            r_squared = linear_regression_r2(joint_df[r'smap_' + str(rchid)], joint_df[r'topnet_' + str(rchid)])
            # view R-squared value
            # print(r_squared)
            my_r2s[i] = r_squared

            if plot_time_series:
                print('plot rchid = ' + str(rchid))
                saveFigName = r'rchid_' + str(rchid) + '_topnet_smap'
                my_fontsize = 14
                year_size = 365  # approx 5 years of daily data

                ax = smap_col_df.plot(marker='.', ms=5, alpha=1, linestyle='None',color='r',
                                      figsize=(5 * (math.ceil(smap_col_df.size / year_size)), 5),
                                      fontsize=my_fontsize, grid=True, label='smap')
                topnet_col_df.plot(ax=ax, label='topnet',color='b')
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
        gdf_reaches['r2_smap_topnet'] = my_r2s
        gdf_reaches.to_pickle(os.path.join(gdf_path, gdf_file + '_r2'))

    if plot_correlation_maps:
        gdf = gdf_reaches.set_geometry("centroid")
        ax = gdf.dropna().plot("r2_smap_topnet", legend=True, markersize=5)

        if my_shape is not None:
            gdf_shape = gpd.read_file(my_shape)
            gdf_shape['boundary'] = gdf_shape.boundary
            gdf_shape['boundary'].plot(ax=ax, color="grey", linewidth=.5)
        # plt.show()
        save_fig_name = 'r2_smap_topnet'
        fig_path = os.path.join(os.getcwd(), r'files\outputs')
        if not os.path.exists(fig_path):
            os.mkdir(fig_path)
        plt.savefig(os.path.join(fig_path, save_fig_name) + '.png', dpi=300)
        plt.close()

    return gdf_reaches


def compare_dataframes_obs_topnet(gdf_obs, df_obs, gdf_reaches, df_topnet, plot_time_series=True, save_new_gdf=False,
                                  plot_correlation_maps=False, my_shape=None):
    import numpy as np
    from libs.modules.utils import linear_regression_r2
    import geopandas as gpd

    input_rchids = list(gdf_obs['Station Rchid'])
    my_r2s = [np.nan] * len(input_rchids)

    gdf_obs_nztm = gdf_obs.to_crs(2193)
    df_obs_list = []
    df_topnet_list = []
    gdf_index_list = []
    for i in range(len(input_rchids)):
        obs_coord = gdf_obs_nztm.iloc[i].geometry
        gdf_in = gdf_reaches.contains(obs_coord)
        if len(gdf_in[gdf_in]) == 1:  # check if not empty
            my_reach = gdf_reaches[gdf_in]
            df_obs_list.append(input_rchids[i])
            df_topnet_list.append(str(my_reach.iloc[0].name))
            gdf_index_list.append(True)
        else:
            gdf_index_list.append(False)

        # d_topnet[str(my_reach.iloc[0].name)] = topnet_df[str(my_reach.iloc[0].name)].copy()

    df_obs2_compare = df_obs[df_obs_list].copy()
    df_obs2_compare.columns = [str(col) + '_obs' for col in df_obs2_compare.columns]
    df_topnet2_compare = df_topnet[df_topnet_list].copy()
    df_topnet2_compare.columns = [str(col) + '_topnet' for col in df_topnet2_compare.columns]
    df_obs2_compare = df_obs2_compare.resample('D').mean()
    df_topnet2_compare = df_topnet2_compare.resample('D').mean()

    my_r2s = [np.nan] * len(df_obs_list)
    for i in range(len(df_obs_list)):
        obs_col = df_obs2_compare[df_obs2_compare.columns[i]]
        topnet_col = df_topnet2_compare[df_topnet2_compare.columns[i]]
        if np.sum(obs_col.count()) > 0 and np.sum(topnet_col.count()) > 0:
            # smap_col_df = smap_col_df.fillna('NaN').astype('float')  # convert NaT to NaN and cast values to float
            # topnet_col_df = topnet_col_df.fillna('NaN').astype('float')  # convert NaT to NaN and cast values to float

            frames = [obs_col, topnet_col]
            joint_df = pd.concat(frames, axis=1)
            joint_df.dropna(inplace=True)
            # joint_df.columns = [r'obs_' + str(rchid), r'topnet_' + str(rchid)]
            # calculate R2 and add to gdfreaches as a column
            # https://www.statology.org/r-squared-in-python/
            # R2 values of:
            # - smap to topnet (done)
            # - use that to interpolate missing smap
            # - smap to field observations(with and without interpolate)
            # - topnet to field observations
            # - smap and topnet to field observations(with and without interpolate)

            # calculate R-squared of regression model
            r_squared = linear_regression_r2(joint_df[df_obs2_compare.columns[i]],
                                             joint_df[df_topnet2_compare.columns[i]])
            # todo: also calculate model fits (in above or similar function)
            # view R-squared value
            # print(r_squared)
            my_r2s[i] = r_squared

            if plot_time_series:
                print('plot obs_station = ' + str(df_obs_list[i]))
                saveFigName = r'obs_id_' + str(df_obs_list[i]) + '_obs_topnet'
                my_fontsize = 14
                year_size = 365  # approx 5 years of daily data

                ax = obs_col.plot(marker='.', ms=5, alpha=1, linestyle='None', color='g',
                                  figsize=(5 * (math.ceil(obs_col.size / year_size)), 5),
                                  fontsize=my_fontsize, grid=True, label='in situ')
                topnet_col.plot(ax=ax, label='topnet',color='b')
                plt.legend(loc='best')
                plt.title(r'soil moisture at obs station id ' + str(df_obs_list[i]), fontsize=my_fontsize)
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

    gdf_obs_nztm['r2_obs_topnet'] = np.nan
    gdf_obs_nztm.loc[gdf_index_list, 'r2_obs_topnet'] = my_r2s

    if save_new_gdf:
        gdf_obs_nztm.to_pickle(os.path.join(os.getcwd(), r'files\dataframes\gdf_obs_topnet_r2'))

    if plot_correlation_maps:
        ax = gdf_obs_nztm.dropna().plot('r2_obs_topnet', legend=True, markersize=10)
        if my_shape is not None:
            gdf_shape = gpd.read_file(my_shape)
            gdf_shape['boundary'] = gdf_shape.boundary
            gdf_shape['boundary'].plot(ax=ax, color="grey", linewidth=.5)
            # plt.show()
        saveFigName = r'r2_obs_topnet'
        fig_path = os.path.join(os.getcwd(), r'files\outputs')
        if not os.path.exists(fig_path):
            os.mkdir(fig_path)
        plt.savefig(os.path.join(fig_path, saveFigName) + '.png', dpi=300)
        plt.close()

    return gdf_obs


def read_smap_at_obs(gdf_obs,
                     data_path=r'i:GroundWater\Research\NIWA_NationalHydrologyProgram\Data\SoilMoistureVanderSat\SmapData\NorthlandNcFiles',
                     buffer=3):
    import numpy as np
    import glob
    import datetime
    from libs.modules.utils import indexContainingSubstring, closestNode, linear_regression_r2
    import netCDF4 as nc
    import maya
    import os
    import geopandas as gpd
    import pandas as pd

    # data_path = r'i:GroundWater\Research\NIWA_NationalHydrologyProgram\Data\SoilMoistureVanderSat\SmapData\NorthlandNcFiles'
    # buffer = 3  # number of SMAP pixels around the field observation to bring into the comparison

    print('reading SMAP data at locations of field observations ...')

    # read smap
    file_list = sorted(glob.glob(os.path.join(data_path, r'SM-SMAP*.nc')))

    # https://www.w3schools.com/python/python_datetime.asp
    start_date = datetime.datetime(2016, 6, 1)
    end_date = datetime.datetime(2021, 5, 31)  # year, month, day

    a = start_date.strftime('%Y%m%d')
    start_date_str = (a[:4] + r'-' + a[4:6] + r'-' + a[6:8])

    a = end_date.strftime('%Y%m%d')
    end_date_str = (a[:4] + r'-' + a[4:6] + r'-' + a[6:8])

    duration_days = (end_date - start_date).days + 1
    # print(str(duration_days) + r' day period')

    index1 = indexContainingSubstring(file_list, start_date_str)
    index2 = indexContainingSubstring(file_list, end_date_str)

    # read coordinates of field observations and find closest smap
    input_obs_station_list = list(gdf_obs['Station Rchid'])

    # gdf_obs['buffered'] = gdf_obs.to_crs(2193).buffer(200)

    d = {}
    for i in range(len(input_obs_station_list)):
        print(r'working on observation: ' + str(i + 1))
        obs_lon = gdf_obs.iloc[i].geometry.x
        obs_lat = gdf_obs.iloc[i].geometry.y

        station_id = str(input_obs_station_list[i])
        column_names = ["Time", station_id]
        d[station_id] = pd.DataFrame(columns=column_names)

        for ifile in range(index1, index2 + 1):

            fn = file_list[ifile]
            ds = nc.Dataset(fn)
            lat = ds['lat'][:]
            lon = ds['lon'][:]
            index_lat = closestNode(obs_lat, lat)  # only used in timeseries
            index_lon = closestNode(obs_lon, lon)  # only used in timeseries
            sm_q = ds['SM-SMAP-L-DESC_V4.0_100'][:, index_lon - buffer:index_lon + buffer + 1,
                   index_lat - buffer:index_lat + buffer + 1]
            mx = np.ma.masked_invalid(sm_q)
            if sum(sum(sum(mx.mask))) == mx.mask.size:
                sm_q = np.nan
            else:
                # df[input_obs_station_list[i]] = sm_q.filled(np.nan).flatten()
                sm_q = sm_q.filled(np.nan).flatten().mean()

            date_time_obj = maya.parse(ds.datetime).datetime()
            d[station_id].loc[ifile] = [date_time_obj, sm_q]  # put data in dataframe

        d[station_id].set_index('Time', inplace=True)

    df_smap_at_obs = pd.concat(d.values(), axis=1)
    return df_smap_at_obs

def compare_smap_at_obs(gdf_obs, df_obs, df_smap_at_obs, plot_time_series=False, plot_correlation_maps=False,
                        my_shape=None, save_new_gdf=True):
    import numpy as np
    import math
    from libs.modules.utils import linear_regression_r2
    import matplotlib.pyplot as plt
    import os
    import geopandas as gpd

    print('Comparing SMAP data to observations...')
    df_obs.columns = [str(col) + '_obs' for col in df_obs.columns]
    df_smap_at_obs.columns = [str(col) + '_smap' for col in df_smap_at_obs.columns]
    df_obs = df_obs.resample('D').mean()
    df_smap_at_obs = df_smap_at_obs.resample('D').mean()

    input_obs_station_list = list(gdf_obs['Station Rchid'])
    my_r2s = [np.nan] * len(input_obs_station_list)
    for i in range(len(input_obs_station_list)):
        obs_col = df_obs[df_obs.columns[i]]  # more generic than df_obs[input_obs_station_list[i]]
        smap_at_obs_col = df_smap_at_obs[df_smap_at_obs.columns[i]]
        if np.sum(obs_col.count()) > 0 and np.sum(smap_at_obs_col.count()) > 0:
            # smap_col_df = smap_col_df.fillna('NaN').astype('float')  # convert NaT to NaN and cast values to float
            # topnet_col_df = topnet_col_df.fillna('NaN').astype('float')  # convert NaT to NaN and cast values to float

            frames = [obs_col, smap_at_obs_col]
            joint_df = pd.concat(frames, axis=1)
            joint_df.dropna(inplace=True)
            # joint_df.columns = [r'obs_' + str(rchid), r'topnet_' + str(rchid)]
            # calculate R2 and add to gdfreaches as a column
            # https://www.statology.org/r-squared-in-python/

            # calculate R-squared of regression model
            r_squared = linear_regression_r2(joint_df[df_obs.columns[i]],
                                             joint_df[df_smap_at_obs.columns[i]])
            # todo: also calculate model fits (in above or similar function)
            # view R-squared value
            # print(r_squared)
            my_r2s[i] = r_squared

            if plot_time_series:
                print('plot obs_station = ' + str(input_obs_station_list[i]))
                saveFigName = r'obs_id_' + str(input_obs_station_list[i]) + '_obs_smap'
                my_fontsize = 14
                year_size = 365  # approx 5 years of daily data

                ax = smap_at_obs_col.plot(marker='.', ms=5, alpha=1, linestyle='None', color='r',
                                          figsize=(5 * (math.ceil(obs_col.size / year_size)), 5),
                                          fontsize=my_fontsize, grid=True, label='smap')
                obs_col.plot(ax=ax, label='in situ', marker='.', ms=5, alpha=1, linestyle='None', color='g')
                plt.legend(loc='best')
                plt.title(r'soil moisture at obs station id ' + str(input_obs_station_list[i]), fontsize=my_fontsize)
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

    gdf_obs['r2_obs_smap'] = my_r2s

    if save_new_gdf:
        gdf_obs.to_pickle(os.path.join(os.getcwd(), r'files\dataframes\gdf_obs_smap_r2'))

    if plot_correlation_maps:
        ax = gdf_obs.to_crs(2193).dropna().plot('r2_obs_smap', legend=True, markersize=10)
        if my_shape is not None:
            gdf_shape = gpd.read_file(my_shape)
            gdf_shape['boundary'] = gdf_shape.boundary
            gdf_shape['boundary'].plot(ax=ax, color="grey", linewidth=.5)
            # plt.show()
        saveFigName = r'r2_obs_smap'
        fig_path = os.path.join(os.getcwd(), r'files\outputs')
        if not os.path.exists(fig_path):
            os.mkdir(fig_path)
        plt.savefig(os.path.join(fig_path, saveFigName) + '.png', dpi=300)
        plt.close()

    return gdf_obs