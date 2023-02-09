import os
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from wrf import getvar, extract_times, to_np, latlon_coords
import cmaps as nclcmaps
from cartopy.feature import NaturalEarthFeature
import cartopy.crs as crs
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from woad import parameter as parm
from woad import actkit
from woad import diagkit


states = NaturalEarthFeature(category='cultural', scale='50m',
                             facecolor='none',
                             name='admin_1_states_provinces_shp')
                                 
def main(infile, outcsv, figName, inilat, inilon, maxwsShRadius=2, angleInterval=5, radiusesEnd=350000, radiusesInterval=1000):
    # to import the csv data, use 'pd.read_csv(outcsv, parse_dates=[0], infer_datetime_format=True)'
    # basic variables
    infile = os.path.abspath(infile)
    print('processing: ' + infile)
    ncfile = Dataset(infile)
    nctime = extract_times(ncfile, 0)
    slp = getvar(ncfile, 'slp')
    u10, v10 = diagkit.uv10(ncfile, mask_height=10000)
    wspd10m = (u10**2+v10**2)**0.5
    lon = np.array(ncfile['XLONG'])[0, :, :]
    lat = np.array(ncfile['XLAT'])[0, :, :]

    # locate tc center by the min-slp
    minslp, minslpLat, minslpLon = actkit.search_extreme_xarray(slp, mode='min', limsh=True, cntLat=inilat, cntLon=inilon, radius=4)

    # locate tc center by the min-slp
    tccLon, tccLat = diagkit.tcc_by_pressureCentroid(to_np(slp), lon, lat, minslpLon, minslpLat, maxIter=100)
    tccslp = actkit.interp_2d_to_point_lonlat_3plinear(to_np(slp), lon, lat, tccLon, tccLat)

    # max ws10
    maxws10, maxws10Lat, maxws10Lon = actkit.search_extreme_xarray(wspd10m, mode='max', limsh=True, cntLat=tccLat, cntLon=tccLon, radius=maxwsShRadius)

    # info diagnosed in cylindrical grid
    rmw, r34, maxAzws10, midproduct = diagkit.tc_wind_InPCS(ncfile=ncfile, wspd=wspd10m, cntLat=tccLat, cntLon=tccLon,
                                                            angleInterval=angleInterval, radiusesEnd=radiusesEnd, radiusesInterval=radiusesInterval)

    # output in csv
    newDF = pd.DataFrame({'time': np.datetime_as_string(nctime, timezone='UTC', unit='s'),
                          'tccslp': '{:9.3f}'.format(tccslp),
                          'tccLat': '{:8.3f}'.format(tccLat),
                          'tccLon': '{:8.3f}'.format(tccLon),
                          'minslp': '{:9.3f}'.format(minslp),
                          'minLat': '{:8.3f}'.format(minslpLat),
                          'minLon': '{:8.3f}'.format(minslpLon),
                          'maxws10': '{:7.3f}'.format(maxws10),
                          'rmw': '{:6.1f}'.format(rmw/1000),
                          'r34': '{:6.1f}'.format(r34/1000),
                          'maxAzws10': '{:7.3f}'.format(maxAzws10),
                          'ncfile': ' '+infile},
                         index=[0])

    if os.path.isfile(outcsv):
        oriDF = pd.read_csv(outcsv, dtype=str)
        newDF = oriDF.append(newDF, ignore_index=True)
    newDF.to_csv(outcsv, index=False, sep=',')

    # plot figure of diagnose result
    if figName.lower() not in parm.denyStr:
        # cut data
        trimLonLat = [tccLon-4.5, tccLon+4.5, tccLat-4, tccLat+4]
        slp_cutted = actkit.trim_data_xarray(slp, trimLonLat=trimLonLat)
        wspd10m_cutted = actkit.trim_data_xarray(wspd10m,  trimLonLat=trimLonLat)

        cart_proj = crs.LambertConformal(central_longitude=tccLon, central_latitude=tccLat)
        cart_proj_polar = crs.AzimuthalEquidistant(central_longitude=tccLon, central_latitude=tccLat)

        fig = plt.figure(figsize=(12, 8))
        ax_slp_ws10m = fig.add_axes([0.07, 0.0, 0.8/1.5, 0.9], projection=cart_proj)
        ax_meanws10 = fig.add_axes([1/1.5, 0.6, 0.4/1.5, 0.3])
        ax_cycoords = fig.add_axes([1/1.5, 0.1, 0.4/1.5, 0.4], projection=cart_proj_polar)

        ax_slp_ws10m.set_title('10m windspeed ans slp', fontsize=14)
        ax_meanws10.set_title('azimuthal mean 10m ws', fontsize=14)
        ax_cycoords.set_title('polor coords', fontsize=14)

        plt_slp_ws10m(ax_slp_ws10m, slp_cutted, wspd10m_cutted, tccLon, tccLat, maxws10Lon, maxws10Lat, rmw, r34, midproduct)
        plt_meanws10_by_dist(ax_meanws10, rmw, r34, maxAzws10, midproduct)
        plt_cycoords(ax_cycoords, midproduct)

        timeinfo = np.datetime_as_string(nctime, timezone='UTC', unit='s')
        plt.text(0.02, 0.96, timeinfo, fontsize=14, transform=plt.gcf().transFigure)
        plt.text(0.02, 0.02, infile, fontsize=10, transform=plt.gcf().transFigure)

        fig.savefig(figName, dpi=150)


def plt_slp_ws10m(ax, slp, wspd10m, tccLon, tccLat, maxws10Lon, maxws10Lat, rmw, r34, midproduct):
    lats, lons = latlon_coords(slp)

    ax.add_feature(states, linewidth=0.5, edgecolor='black')
    ax.coastlines('50m', linewidth=0.8)

    # plot contourf of wpsd10m
    levels = np.arange(10, 60.5, 0.5)
    wspd_contours = ax.contourf(to_np(lons), to_np(lats), to_np(wspd10m),
                                levels=levels,
                                cmap=nclcmaps.WhiteBlueGreenYellowRed,
                                transform=crs.PlateCarree())
    cbar = plt.colorbar(wspd_contours, ax=ax, orientation='horizontal', pad=.05)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_ticks(ticks=mticker.FixedLocator(range(10, 65, 5)), update_ticks=True)

    # plot contour of slp
    levels = np.arange(850, 1050, 10)
    ax.contour(to_np(lons), to_np(lats), to_np(slp),
               colors='b', alpha=0.5,
               levels=levels, transform=crs.PlateCarree())
    levels = np.arange(855, 1055, 10)
    ax.contour(to_np(lons), to_np(lats), to_np(slp),
               colors='b', alpha=0.3,
               levels=levels, transform=crs.PlateCarree())

    # plot tc center and rmw
    ax.plot(tccLon, tccLat, 'xk', transform=crs.PlateCarree())
    ax.scatter(maxws10Lon, maxws10Lat, color='none', edgecolor='black', transform=crs.PlateCarree())

    # adjust range of lon lat
    ax.set_extent([tccLon-3.5, tccLon+3.5, tccLat-3, tccLat+3], crs=crs.PlateCarree())

    gl = ax.gridlines(crs=crs.PlateCarree(), alpha=0.5,
                      linestyle='--', draw_labels=True,
                      x_inline=False, y_inline=False)
    gl.top_labels = False
    gl.right_labels = False

    # plot cirecle of RMW
    idx = midproduct['radiuses'] == rmw
    rmw_lon = midproduct['lonInPCS'][:, idx]
    rmw_lat = midproduct['latInPCS'][:, idx]
    rmw_lon = np.concatenate((rmw_lon[:, 0], rmw_lon[0]), axis=None)
    rmw_lat = np.concatenate((rmw_lat[:, 0], rmw_lat[0]), axis=None)
    ax.plot(rmw_lon, rmw_lat,
            color=[0.0, 0.0, 0.0], alpha=0.5, transform=crs.PlateCarree())

    # plot cirecle of R34
    if r34 > 0:
        r34_lat = np.empty([len(midproduct['angles'])+1])
        r34_lon = np.empty([len(midproduct['angles'])+1])
        for iangle in np.arange(0, len(midproduct['angles'])):
            intpfunc = interp1d(midproduct['radiuses'],  midproduct['lonInPCS'][iangle, :])
            r34_lon[iangle] = intpfunc(r34)
            intpfunc = interp1d(midproduct['radiuses'],  midproduct['latInPCS'][iangle, :])
            r34_lat[iangle] = intpfunc(r34)

        r34_lat[-1] = r34_lat[0]
        r34_lon[-1] = r34_lon[0]
        ax.plot(r34_lon, r34_lat,
                color=[0.0, 0.0, 0.0], alpha=0.5, transform=crs.PlateCarree())


def plt_meanws10_by_dist(ax, rmw, r34, maxAzws10, midproduct):
    #
    xEnd = midproduct['radiuses'][-1]/1000
    ax.plot([0, xEnd], [34*parm.knot2ms, 34*parm.knot2ms], linestyle='--', color='#3c79c8')
    ax.plot([0, xEnd], [64*parm.knot2ms, 64*parm.knot2ms], linestyle='--', color='#3ce682')
    ax.plot([0, xEnd], [100*parm.knot2ms, 100*parm.knot2ms], linestyle='--', color='#e6323b')
    ax.text(xEnd*0.8, 36*parm.knot2ms, '34 kts', color='#3c79c8')
    ax.text(xEnd*0.8, 66*parm.knot2ms, '64 kts', color='#3ce682')
    ax.text(xEnd*0.8, 102*parm.knot2ms, '100 kts', color='#e6323b')

    #
    if r34 > 0:
        ax.plot([r34, r34], [0, 2], 'k')
        ax.text(r34, 3.5, '{:d}'.format(r34), 'k', horizontalalignment='center')

    ax.plot([rmw, rmw], [0, 2], 'k')
    ax.text(rmw, 3.5, '{:d}'.format(rmw), 'k', horizontalalignment='center')

    ax.plot([-5, 0], [maxAzws10, maxAzws10], 'k')
    ax.text(0, maxAzws10, '{:.1f}'.format(maxAzws10), 'k', horizontalalignment='left')

    #
    ax.plot(midproduct['radiuses']/1000, np.min(midproduct['wsInPCS'], axis=0), color='#969696')
    ax.plot(midproduct['radiuses']/1000, np.max(midproduct['wsInPCS'], axis=0), color='#969696')
    ax.plot(midproduct['radiuses']/1000, np.mean(midproduct['wsInPCS'], axis=0), color='black')

    ax.set_xlabel('distance (km)', fontsize=12)
    ax.set_ylabel('windspeed (m/s)', fontsize=12)
    ax.grid(alpha=0.5, linestyle='--')

    ax.set_xlim(-5, xEnd+5)
    ax.set_ylim(0, 60)
    ax.set_xticks(np.arange(0, xEnd+5, 50))
    ax.set_yticks(np.arange(0, 70, 5))


def plt_cycoords(ax, midproduct):
    states = NaturalEarthFeature(category='cultural', scale='50m',
                                 facecolor='none',
                                 name='admin_1_states_provinces_shp')
    ax.add_feature(states, linewidth=0.5, edgecolor='black')
    ax.coastlines('50m', linewidth=0.8)

    idx = midproduct['wsInPCS'] <= 34*parm.knot2ms
    ax.scatter(midproduct['lonInPCS'][idx], midproduct['latInPCS'][idx],
               s=0.5, c='gray', edgecolor=None, transform=crs.PlateCarree())
    idx = np.logical_and(midproduct['wsInPCS'] >= 34*parm.knot2ms, midproduct['wsInPCS'] < 64*parm.knot2ms)
    ax.scatter(midproduct['lonInPCS'][idx], midproduct['latInPCS'][idx],
               s=0.5, c='#3c79c8', edgecolor=None, transform=crs.PlateCarree())
    idx = np.logical_and(midproduct['wsInPCS'] >= 64*parm.knot2ms, midproduct['wsInPCS'] < 100*parm.knot2ms)
    ax.scatter(midproduct['lonInPCS'][idx], midproduct['latInPCS'][idx],
               s=0.5, c='#3ce682', edgecolor=None, transform=crs.PlateCarree())
    idx = midproduct['wsInPCS'] >= 100*parm.knot2ms
    ax.scatter(midproduct['lonInPCS'][idx], midproduct['latInPCS'][idx],
               s=0.5, c='#e6323b', edgecolor=None, transform=crs.PlateCarree())

    gl = ax.gridlines(crs=crs.PlateCarree(), alpha=0.5,
                      linestyle='--', draw_labels=True,
                      x_inline=False, y_inline=False)
    gl.top_labels = False
    gl.right_labels = False
