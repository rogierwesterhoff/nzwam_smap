# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 11:10:44 2022

@author: rogierw
"""

import numpy as np
import netCDF4 as nc
from matplotlib import path


def indexContainingSubstring(the_list, substring):
    for i, s in enumerate(the_list):
        if substring in s:
              return i
    return -1

def closestNode(node, nodes):
    nodes = np.asarray(nodes)
    deltas = nodes - node
    return np.argmin(abs(deltas))

def getScreenWidth():
    from win32api import GetSystemMetrics
    screen_width = GetSystemMetrics(0)
    return screen_width

def getScreenHeight():
    from win32api import GetSystemMetrics
    screen_height = GetSystemMetrics(1)
    return screen_height

def movingaverage(interval, window_size):
    import numpy
    window= numpy.ones(int(window_size))/float(window_size)
    return numpy.convolve(interval, window, 'same')

def find_nearest_value(array, value):
    '''
    finds nearest value
    :param array:
    :param value:
    :return:
    '''
    import numpy as np
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def find_nearest_index(array, value):
    '''
    finds nearest index to value in array
    :param array:
    :param value:
    :return:
    '''
    import numpy as np
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

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

def inpolygon(xq, yq, xv, yv):
    """
    :param xq: x coordinate of the query point
    :param yq: y coordinate of the query point
    :param xv: x coordinates of the shape
    :param yv: y coordinates of the shape
    :return: x, y coordinates that are in the shape (not on)
    """
    shape = xq.shape
    xq = xq.reshape(-1)
    yq = yq.reshape(-1)
    xv = xv.reshape(-1)
    yv = yv.reshape(-1)
    q = [(xq[i], yq[i]) for i in range(xq.shape[0])]
    p = path.Path([(xv[i], yv[i]) for i in range(xv.shape[0])])
    return p.contains_points(q).reshape(shape)