import numpy as np
import matplotlib.pyplot as plt
import os
import netCDF4 as nc
import glob
import datetime
import time
import pandas as pd


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

    # todo: check whether geopandas is needed

    from libs.modules.utils import indexContainingSubstring, closestNode

    # +++++++ END OF FUNCTIONS +++++
    t = time.time()

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
        # todo: check if these could also be a range (e.g. all indices within a polygon)
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

    elapsed = time.time() - t
    print(r'run time: ' + str(round(elapsed) / 60) + r' minutes')
    print(r'Dataframe pickled in: ' + df_filename)

def convert_nc_time(ds, tname):
    '''
    Converts numeric netcdf time data to python datetime.

    Parameters
    ----------
    ds: dataset from netcdf file as read in by ds = nc.Dataset(fn)
    time: (str). ds column name that contains time information (eg. 'time')

    Returns
    -------
    times: timeseries in python datetime
    '''

    nc_time = ds.variables[tname][:]
    t_unit = ds.variables[tname].units

    py_times = nc.num2date(nc_time[:].squeeze(), t_unit,
                     only_use_cftime_datetimes=False,
                     only_use_python_datetimes=True)

    return py_times