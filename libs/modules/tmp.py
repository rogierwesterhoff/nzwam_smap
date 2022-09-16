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