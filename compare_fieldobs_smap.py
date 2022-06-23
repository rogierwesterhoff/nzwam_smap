smap_df
topnet_df
gdf_path
gdf_file

plot_time_series = False
save_new_gdf = False
plot_correlation_maps = False

import matplotlib.pyplot as plt
import numpy as np
import math
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