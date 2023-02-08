import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import numpy as np
from wrf import getvar, latlon_coords, extract_times
from woad import diagkit, plotkit, actkit, mathkit
import cartopy.crs as crs
import cmaps as nclcmaps


def plot_contourf_skip_zero(ax, var, lons, lats, cLevInterval=None, cbarTicks=None, vmax=None, pltcbar=True, pltbdy=True):
    if vmax is None:
        vmax = np.max(np.abs(var))

    climMax = mathkit.ceil_nthDigit(vmax, n=2)
    midnorm = plotkit.MidpointNormalize(vmin=-climMax, vcenter=0, vmax=climMax)

    if cLevInterval is None:
        cLevInterval = mathkit.ceil_nthDigit(climMax/6, n=2)
        if cLevInterval < 1:
            cLevInterval = mathkit.ceil_nthDigit(climMax/6, n=1)

    temp = np.arange(cLevInterval, climMax, cLevInterval)
    levels = np.sort(np.unique(np.concatenate([-temp, temp, [-climMax, climMax]])))
    levels = levels[levels != 0]

    #
    contours_0 = ax.contourf(lons, lats, var,
                             levels=levels[levels < 0],
                             vmin=-climMax, vmax=climMax,
                             cmap='bwr', norm=midnorm,
                             extend='min',
                             transform=crs.PlateCarree())

    contours_1 = ax.contourf(lons, lats, var,
                             levels=levels[levels > 0],
                             vmin=-climMax, vmax=climMax,
                             cmap='bwr', norm=midnorm,
                             extend='max',
                             transform=crs.PlateCarree())

    if pltcbar:
        if cbarTicks is None:
            cbarTicks = levels
        contours_for_cbar = ax.contourf(lons[0:2, 0:2], lats[0:2, 0:2],
                                        np.array([[-climMax, climMax], [climMax, -climMax]]),
                                        levels=levels,
                                        vmin=-climMax, vmax=climMax,
                                        cmap='bwr', norm=midnorm,
                                        transform=crs.PlateCarree())
        cbar = plt.colorbar(contours_for_cbar, ax=ax, pad=.05, ticks=levels)
        cbar.ax.tick_params(labelsize=12)
        cbar.set_ticks(ticks=cbarTicks, update_ticks=True)
    else:
        cbar = None

    if pltbdy:
        ax.plot(lons[0, :], lats[0, :], linestyle='dashed', linewidth=2, color=[0.75, 0.75, 0.75], alpha=0.6, transform=crs.PlateCarree())
        ax.plot(lons[-1, :], lats[-1, :], linestyle='dashed', linewidth=2, color=[0.75, 0.75, 0.75], alpha=0.6, transform=crs.PlateCarree())
        ax.plot(lons[:, 0], lats[:, 0], linestyle='dashed', linewidth=2, color=[0.75, 0.75, 0.75], alpha=0.6, transform=crs.PlateCarree())
        ax.plot(lons[:, -1], lats[:, -1], linestyle='dashed', linewidth=2, color=[0.75, 0.75, 0.75], alpha=0.6, transform=crs.PlateCarree())

    #
    ax.set_extent([np.min(lons), np.max(lons), np.min(lats), np.max(lats)])

    #
    gl = ax.gridlines(crs=crs.PlateCarree(), alpha=0.5,
                      linestyle='--', draw_labels=True,
                      x_inline=False, y_inline=False)
    gl.top_labels = False
    gl.right_labels = False

    #
    pltObj = {"diff_contours_positive": contours_1,
              "diff_contours_negative": contours_0,
              "cbar": cbar,
              "gl": gl}

    return pltObj


def plot_cloudTopTemp_gray(ax, ncfile, trimLonLat=None, plotcbar=False):
    ctt = getvar(ncfile, 'ctt', units='degC')
    if trimLonLat is not None:
        ctt = actkit.trim_data_xarray(ctt, trimLonLat=trimLonLat)

    lats, lons = latlon_coords(ctt)

    clevels = np.arange(-80, 20, 10)
    ctt_contourf = ax.contourf(lons, lats, ctt,
                               cmap=get_cmap("Greys"), levels=clevels, extend='min',
                               vmin=-80, vmax=20,
                               transform=crs.PlateCarree())

    if plotcbar:
        plt.colorbar(ctt_contourf, ax=ax, ticks=clevels)

    gl = ax.gridlines(crs=crs.PlateCarree(), alpha=0.5,
                      linestyle='--', draw_labels=True,
                      x_inline=False, y_inline=False)
    gl.top_labels = False
    gl.right_labels = False

    return ctt_contourf, gl


def plot_cloudTopTemp_enhanced(ax, ncfile, trimLonLat=None, plotcbar=False):
    ctt = getvar(ncfile, 'ctt', units='degC')
    if trimLonLat is not None:
        ctt = actkit.trim_data_xarray(ctt, trimLonLat=trimLonLat)

    lats, lons = latlon_coords(ctt)

    cmp, norm = plotkit.cmap_cwb_satellite_ctt_enhanced()
    cbar_nodes = [28, 23, 0, -20, -25, -31, -42, -53, -59, -63, -80, -110, -136]
    clevels = np.concatenate([cbar_nodes, np.arange(-20, 28), np.arange(-80, -63), np.arange(-136, -110)])
    clevels = np.sort(np.unique(clevels))
    ctt_contourf = ax.contourf(lons, lats, ctt,
                               cmap=cmp, levels=clevels, extend='both', norm=norm,
                               transform=crs.PlateCarree())

    if plotcbar:
        plt.colorbar(ctt_contourf, ax=ax, ticks=cbar_nodes)

    gl = ax.gridlines(crs=crs.PlateCarree(), alpha=0.5,
                      linestyle='--', draw_labels=True,
                      x_inline=False, y_inline=False)
    gl.top_labels = False
    gl.right_labels = False

    return ctt_contourf, gl


def plot_reflectivity(ax, ncfile, trimLonLat=None, plotcbar=False):
    reflect = getvar(ncfile, 'mdbz')
    if trimLonLat is not None:
        reflect = actkit.trim_data_xarray(reflect, trimLonLat=trimLonLat)

    lats, lons = latlon_coords(reflect)
    cmp, norm = plotkit.cmap_cwb_radar_reflec()

    clevels = np.arange(0, 66, 1)
    dbz_contourf = ax.contourf(lons, lats, reflect,
                               cmap=cmp, levels=clevels, norm=norm, extend='max',
                               transform=crs.PlateCarree())

    if plotcbar:
        plt.colorbar(dbz_contourf, ax=ax, ticks=np.arange(0, 66, 5))

    gl = ax.gridlines(crs=crs.PlateCarree(), alpha=0.5,
                      linestyle='--', draw_labels=True,
                      x_inline=False, y_inline=False)
    gl.top_labels = False
    gl.right_labels = False

    return dbz_contourf, gl


def plot_rainfallRate(ax, ncfileStart, ncfileEnd, trimLonLat=None, plotcbar=False):
    lons = np.array(ncfileStart['XLONG'])[0, :, :]
    lats = np.array(ncfileStart['XLAT'])[0, :, :]
    accRainfallStart = (np.array(ncfileStart['RAINNC'])+np.array(ncfileStart['RAINC']))[0, :, :]
    accRainfallEnd = (np.array(ncfileEnd['RAINNC'])+np.array(ncfileEnd['RAINC']))[0, :, :]

    if trimLonLat is not None:
        idxMinX, idxMaxX, idxMinY, idxMaxY = actkit.cal_trimed_lonlat_idx(lons, lats, trimLonLat)
        lons = lons[idxMinX:idxMaxX, idxMinY:idxMaxY]
        lats = lats[idxMinX:idxMaxX, idxMinY:idxMaxY]
        accRainfallStart = accRainfallStart[idxMinX:idxMaxX, idxMinY:idxMaxY]
        accRainfallEnd = accRainfallEnd[idxMinX: idxMaxX, idxMinY: idxMaxY]

    timeLen = extract_times(ncfileEnd, 0)-extract_times(ncfileStart, 0)
    rainfallRate = (accRainfallEnd-accRainfallStart)*(np.timedelta64(1, 'h')/timeLen)

    cmp, norm = plotkit.cmap_jaxa_rainfall_rate()
    clevels = np.array([0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
    rfr_contourf = ax.contourf(lons, lats, rainfallRate,
                               cmap=cmp, levels=clevels, norm=norm, extend='max',
                               transform=crs.PlateCarree())
    if plotcbar:
        plt.colorbar(rfr_contourf, ax=ax, ticks=clevels)

    gl = ax.gridlines(crs=crs.PlateCarree(), alpha=0.5,
                      linestyle='--', draw_labels=True,
                      x_inline=False, y_inline=False)
    gl.top_labels = False
    gl.right_labels = False

    return rfr_contourf, gl


def plot_accRainfall(ax, ncfileEnd, ncfileStart=None, trimLonLat=None, plotcbar=False):
    lons = np.array(ncfileEnd['XLONG'])[0, :, :]
    lats = np.array(ncfileEnd['XLAT'])[0, :, :]

    if ncfileStart is not None:
        accRainfall = (np.array(ncfileEnd['RAINNC'])+np.array(ncfileEnd['RAINC']))[0, :, :] - (np.array(ncfileStart['RAINNC'])+np.array(ncfileStart['RAINC']))[0, :, :]
    else:
        accRainfall = (np.array(ncfileEnd['RAINNC'])+np.array(ncfileEnd['RAINC']))[0, :, :]

    if trimLonLat is not None:
        idxMinX, idxMaxX, idxMinY, idxMaxY = actkit.cal_trimed_lonlat_idx(lons, lats, trimLonLat)
        lons = lons[idxMinX:idxMaxX, idxMinY:idxMaxY]
        lats = lats[idxMinX:idxMaxX, idxMinY:idxMaxY]
        accRainfall = accRainfall[idxMinX:idxMaxX, idxMinY:idxMaxY]

    cmp, norm = plotkit.cmap_cwb_rainfall()
    clevels = np.array([0, 1, 2, 6, 10, 15, 20, 30, 40, 50, 70, 90, 110, 130, 150, 200, 300])
    rfr_contourf = ax.contourf(lons, lats, accRainfall,
                               cmap=cmp, levels=clevels, norm=norm, extend='both',
                               transform=crs.PlateCarree())
    if plotcbar:
        plt.colorbar(rfr_contourf, ax=ax, ticks=clevels)

    gl = ax.gridlines(crs=crs.PlateCarree(), alpha=0.5,
                      linestyle='--', draw_labels=True,
                      x_inline=False, y_inline=False)
    gl.top_labels = False
    gl.right_labels = False

    return rfr_contourf, gl


def plot_rate_slp_blowThreshold(ax, ncfiles, threshold, trimLonLat=None, plotcbar=False):
    lons = np.array(ncfiles[0]['XLONG'])[0, :, :]
    lats = np.array(ncfiles[0]['XLAT'])[0, :, :]
    probability = np.zeros(lons.shape)
    for iNCfile in ncfiles:
        slp = getvar(iNCfile, 'slp')
        temp = slp < threshold
        probability = probability+temp

    probability = probability/len(ncfiles)*100

    if trimLonLat is not None:
        idxMinX, idxMaxX, idxMinY, idxMaxY = actkit.cal_trimed_lonlat_idx(lons, lats, trimLonLat)
        lons = lons[idxMinX:idxMaxX, idxMinY:idxMaxY]
        lats = lats[idxMinX:idxMaxX, idxMinY:idxMaxY]
        probability = probability[idxMinX:idxMaxX, idxMinY:idxMaxY]

    clevels = np.arange(0, 101, 10)
    probability_contourf = ax.contourf(lons, lats, probability, levels=clevels,
                                       cmap=nclcmaps.WhiteBlueGreenYellowRed,
                                       transform=crs.PlateCarree())

    if plotcbar:
        plt.colorbar(probability_contourf, ax=ax, ticks=clevels)

    gl = ax.gridlines(crs=crs.PlateCarree(), alpha=0.5,
                      linestyle='--', draw_labels=True,
                      x_inline=False, y_inline=False)
    gl.top_labels = False
    gl.right_labels = False

    return probability_contourf, gl


def plot_rate_10mWS_aboveThreshold(ax, ncfiles, threshold, trimLonLat=None, plotcbar=False):
    lons = np.array(ncfiles[0]['XLONG'])[0, :, :]
    lats = np.array(ncfiles[0]['XLAT'])[0, :, :]
    probability = np.zeros(lons.shape)
    for iNCfile in ncfiles:
        print("prcessing: wspd10m")
        u10, v10 = diagkit.uv10(iNCfile, mask_height=10000)
        wspd10m = (u10**2+v10**2)**0.5
        temp = wspd10m > threshold
        probability = probability+temp

    probability = probability/len(ncfiles)*100

    if trimLonLat is not None:
        idxMinX, idxMaxX, idxMinY, idxMaxY = actkit.cal_trimed_lonlat_idx(lons, lats, trimLonLat)
        lons = lons[idxMinX:idxMaxX, idxMinY:idxMaxY]
        lats = lats[idxMinX:idxMaxX, idxMinY:idxMaxY]
        probability = probability[idxMinX:idxMaxX, idxMinY:idxMaxY]

    clevels = np.arange(0, 101, 10)
    probability_contourf = ax.contourf(lons, lats, probability, levels=clevels,
                                       cmap=nclcmaps.WhiteBlueGreenYellowRed,
                                       transform=crs.PlateCarree())

    if plotcbar:
        plt.colorbar(probability_contourf, ax=ax, ticks=clevels)

    gl = ax.gridlines(crs=crs.PlateCarree(), alpha=0.5,
                      linestyle='--', draw_labels=True,
                      x_inline=False, y_inline=False)
    gl.top_labels = False
    gl.right_labels = False

    return probability_contourf, gl

# def plot_lowlayer_synoptic():
# def plot_midlayer_synoptic():
# def plot_p_u_v(ncfile):
# def plot_p_pv():
# def plot_rh():
# def plot_precipitableWater():
# def plot_vertCross_uvSpeed_w():
# def plot_vertCross_theta():
# def plot_vertCross_reflectivity():
