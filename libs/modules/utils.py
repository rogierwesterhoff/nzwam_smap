# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 11:10:44 2022

@author: rogierw
"""

def indexContainingSubstring(the_list, substring):
    for i, s in enumerate(the_list):
        if substring in s:
              return i
    return -1

def closestNode(node, nodes):
    import numpy as np
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
    import numpy as np
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

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

    import netCDF4 as nc

    nc_time = ds.variables[tname][:]
    t_unit = ds.variables[tname].units

    py_times = nc.num2date(nc_time[:].squeeze(), t_unit,
                     only_use_cftime_datetimes=False,
                     only_use_python_datetimes=True)

    return py_times

def do_lreg(df):
    from sklearn.linear_model import LinearRegression
    sample = df.sample(df.shape[0], replace=True)
    X_tr = sample[[c for c in sample.columns if c != 'y']]
    y_tr = sample.y

    lr = LinearRegression()
    lr.fit(X_tr, y_tr)

    params = [lr.intercept_] + list(lr.coef_)

    return params

def linear_regression_r2(X,y, output_all = False):
    '''
    calculates R-squared using linear regression. Expects dataframe input
    :param X: predictor variables, e.g., df[["hours", "prep_exams"]]. Can be np array or pd dataframe/series
    :param y: response variable, e.g. df.score
    :return: r_squared (float), (if output_all = True) coefficients of fit (dataframe)
    # background https://www.statology.org/r-squared-in-python/
    # And https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

    If output_all = True, you can call this function in two ways:
    1. output = linear_regresssion(X,y, output_all = True): the output is a tuple of r2 (float) and coeffs (dataframe)
    1. r2s, coeffs = linear_regression(X,y, output_all = True): the outputs are a float (r2) and a dataframe (coeffs)
    '''

    from sklearn.linear_model import LinearRegression
    import pandas as pd
    import numpy as np
    import scipy.stats

    if type(X).__module__ == np.__name__ and a.ndim == 1:
        X = X.reshape(-1, 1)

    if isinstance(X, pd.Series): # if 1D, convert to 2D
        X = np.array(X.values.tolist()).reshape((-1, 1))

    if isinstance(y, pd.Series): # if 1D, convert to 2D
        y = np.array(y.values.tolist()).reshape((-1, 1))

    # initiate linear regression model
    model = LinearRegression()

    # fit regression model
    model.fit(X, y)

    # calculate R-squared of regression model
    r_squared = model.score(X, y)

    if output_all:
        df = pd.DataFrame(X)
        df['y'] = y

        r_df = pd.DataFrame([do_lreg(df) for _ in range(100)])

        w = [model.intercept_[0]] + list(model.coef_[0]) # todo: but what if coef has more than one component?
        # w = [model.intercept_] + list(model.coef_)
        se = r_df.std()

        dof = X.shape[0] - X.shape[1] - 1

        summary = pd.DataFrame({
            'w': w,
            'se': se,
            't': w / se,
            '.025': w - se,
            '.975': w + se,
            'df': [dof for _ in range(len(w))]
        })

        summary['P>|t|'] = scipy.stats.t.sf(abs(summary.t), df=summary.df)
        return r_squared, summary
    else:
        return r_squared