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
    smplot = plt.imshow(data, cmap=plt.cm.terrain_r, interpolation='none', \
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