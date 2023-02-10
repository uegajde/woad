# %%
import sys
import numpy as np
from netCDF4 import Dataset
from wrf import getvar, to_np, vinterp
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as crs
import cmaps as nclcmaps
from woad import parameter as parm
from woad import dpkit
from woad import diagkit
from woad import plotkit

infile = sys.argv[1]
figName = sys.argv[2]
inilat = float(sys.argv[3])
inilon = float(sys.argv[4])
ncfile = Dataset(infile)
lon = np.array(ncfile['XLONG'])[0, :, :]
lat = np.array(ncfile['XLAT'])[0, :, :]
plot_diagnostic = True

tccShRadius = 4  # unit: degree of lat/lon
maxwsShRadius = 2  # unit: degree of lat/lon
angleInterval = 5  # unit: degree
radiusesEnd = 300000  # unit: m
radiusesInterval = 2000  # unit: m

u = getvar(ncfile, 'ua')
v = getvar(ncfile, 'va')
p = getvar(ncfile, 'pressure')
z = getvar(ncfile, 'height')
slp = getvar(ncfile, 'slp')

tccLon, tccLat = diagkit.tcc_by_pressureCentroid(to_np(slp), lon, lat, inilon, inilat, maxIter=100)
cart_proj = crs.LambertConformal(central_longitude=tccLon, central_latitude=tccLat)

radiuses = np.arange(0, 302000, 2000)
plevels = np.arange(np.floor(1000/10)*10, 99, -10)

g = parm.gravitational_acceleration(tccLat)
f = parm.coriolis_coefficient(tccLat)
angles = np.arange(0, 360, angleInterval)
radiuses = np.arange(0, radiusesEnd+radiusesInterval, radiusesInterval)
radiusesForZ = np.arange(0, radiusesEnd+radiusesInterval*2, radiusesInterval)

uInPCS, _, _ = dpkit.interp_to_polarCoord3D_xarray(var=u, zVar=p, ncfile=ncfile,
                                                   cntLat=tccLat, cntLon=tccLon,
                                                   angles=angles, radiuses=radiuses, zlevels=plevels)
vInPCS, _, _ = dpkit.interp_to_polarCoord3D_xarray(var=v, zVar=p, ncfile=ncfile,
                                                   cntLat=tccLat, cntLon=tccLon,
                                                   angles=angles, radiuses=radiuses, zlevels=plevels)
zInPCS, _, _ = dpkit.interp_to_polarCoord3D_xarray(var=z, zVar=p, ncfile=ncfile,
                                                   cntLat=tccLat, cntLon=tccLon,
                                                   angles=angles, radiuses=radiusesForZ, zlevels=plevels)

tanWind, _ = diagkit.tanRadWind_InPCS(uInPCS, vInPCS, angles)
tanWindAzMean = np.mean(tanWind, 1)
zAzMean = np.mean(zInPCS, 1)

term1 = np.zeros(tanWindAzMean.shape)
term1temp = -g*(zAzMean[:, 2:]-zAzMean[:, :-2])/2/radiusesInterval
term1[:, 1:] = term1temp

term2 = np.empty(tanWindAzMean.shape)
term2 = np.mean(tanWind**2, axis=1)/radiuses
term2[:, 0] = 0

term3 = f*tanWindAzMean

netRadialForce = term1+term2+term3

# %%


def plot_force_radiusHeight(ax, datain, zmax=None, zinterval=0.5):
    if zmax is None:
        tenkmidx = radiuses >= 10*1000
        zmax = np.nanmax(np.abs(datain*3600)[:, tenkmidx])
        zmax = np.ceil(zmax/10)*10

    if zmax < 50:
        levinterval = 5
    elif zmax < 100:
        levinterval = 10
    else:
        levinterval = 20

    levels = np.arange(-zmax, zmax+zinterval, zinterval)
    midnorm = plotkit.MidpointNormalize(vmin=-zmax, vcenter=0, vmax=zmax)

    smth_data = dpkit.smooth_2DArray(datain, cntWeight=1, rndWeight=1, loop=1)

    contourfh = ax.contourf(radiuses/1000, plevels, smth_data*3600, levels=levels, cmap='bwr', norm=midnorm)
    cbar = plt.colorbar(contourfh, ax=ax)
    cbar.set_ticks(ticks=mticker.FixedLocator(np.arange(-zmax, zmax+levinterval, levinterval)), update_ticks=True)

    levels = np.arange(levinterval, zmax+levinterval, levinterval)
    ax.contour(radiuses/1000, plevels, smth_data*3600,
               levels=levels, colors='gray')
    levels = np.arange(-zmax, 0, levinterval)
    ax.contour(radiuses/1000, plevels, smth_data*3600, '--',
               levels=levels, colors='gray')
    ax.invert_yaxis()


# %%


def plt_tand_rad_wind(ax, u, v, minusMean):
    angles = np.arange(0, 360, 5)
    radiuses = np.arange(0, radiusesEnd+radiusesInterval, radiusesInterval)

    uInPCS, latInPCS, lonInPCS = dpkit.interp_to_polarCoord2D_xarray(u, ncfile, tccLat, tccLon, angles, radiuses)
    vInPCS, _, _ = dpkit.interp_to_polarCoord2D_xarray(v, ncfile, tccLat, tccLon, angles, radiuses)

    if minusMean:
        uInPCS = uInPCS-np.nanmean(uInPCS)
        vInPCS = vInPCS-np.nanmean(vInPCS)

    tanWind, radWind = diagkit.tanRadWind_InPCS(uInPCS, vInPCS, angles)
    tanWind = np.concatenate([np.reshape(tanWind[-1, :], [1, len(tanWind[-1, :])]), tanWind])
    radWind = np.concatenate([np.reshape(radWind[-1, :], [1, len(radWind[-1, :])]), radWind])
    latInPCS = np.concatenate([np.reshape(latInPCS[-1, :], [1, len(latInPCS[-1, :])]), latInPCS])
    lonInPCS = np.concatenate([np.reshape(lonInPCS[-1, :], [1, len(lonInPCS[-1, :])]), lonInPCS])

    # plot contourf of tanWind
    levels = np.arange(10, 67, 3)
    ax.contourf(lonInPCS, latInPCS, tanWind,
                levels=levels,
                cmap=nclcmaps.WhiteBlueGreenYellowRed,
                transform=crs.PlateCarree())

    # contour of radius
    ax.plot(tccLon, tccLat, 'xk', transform=crs.PlateCarree())
    for irad in np.arange(100, 600, 100):
        idx = np.where(radiuses == irad*1000)
        if len(idx[0]) == 1:
            idx = idx[0][0]
            ax.plot(lonInPCS[:, idx], latInPCS[:, idx],
                    linewidth=4,
                    color=[0.5, 0.5, 0.5], alpha=0.2, transform=crs.PlateCarree())
    ax.plot(lonInPCS[:, -1], latInPCS[:, -1],
            linewidth=4,
            color=[0.5, 0.5, 0.5], alpha=0.7, transform=crs.PlateCarree())

    # contour of radWind
    interval = 5
    levels = np.arange(-interval*10, 0, interval)
    ax.contour(lonInPCS, latInPCS, radWind,
               colors="black", alpha=0.8, linestyles="dashed",
               levels=levels, transform=crs.PlateCarree())
    levels = np.arange(interval, interval*11, interval)
    ax.contour(lonInPCS, latInPCS, radWind,
               colors="black", alpha=0.8,
               levels=levels, transform=crs.PlateCarree())


def plot_z(ax, z):
    angles = np.arange(0, 360, 5)
    radiuses = np.arange(0, radiusesEnd+radiusesInterval, radiusesInterval)

    zInPCS, latInPCS, lonInPCS = dpkit.interp_to_polarCoord2D_xarray(z, ncfile, tccLat, tccLon, angles, radiuses)
    zInPCS = zInPCS-zInPCS[0, 0]
    zInPCS = np.concatenate([np.reshape(zInPCS[-1, :], [1, len(zInPCS[-1, :])]), zInPCS])
    latInPCS = np.concatenate([np.reshape(latInPCS[-1, :], [1, len(latInPCS[-1, :])]), latInPCS])
    lonInPCS = np.concatenate([np.reshape(lonInPCS[-1, :], [1, len(lonInPCS[-1, :])]), lonInPCS])

    # plot contourf of wpsd10m
    midnorm = plotkit.MidpointNormalize(vmin=-500.0, vcenter=0, vmax=500.0)
    levels = np.arange(-15-30*15, 15+30*16, 30)
    ax.contourf(lonInPCS, latInPCS, zInPCS,
                levels=levels, cmap='seismic_r', norm=midnorm,
                transform=crs.PlateCarree())

    # contour of radius
    ax.plot(tccLon, tccLat, 'xk', transform=crs.PlateCarree())
    for irad in np.arange(100, 600, 100):
        idx = np.where(radiuses == irad*1000)
        if len(idx[0]) == 1:
            idx = idx[0][0]
            ax.plot(lonInPCS[:, idx], latInPCS[:, idx],
                    linewidth=4,
                    color=[0.5, 0.5, 0.5], alpha=0.2, transform=crs.PlateCarree())
    ax.plot(lonInPCS[:, -1], latInPCS[:, -1],
            linewidth=4,
            color=[0.5, 0.5, 0.5], alpha=0.7, transform=crs.PlateCarree())


# %% diagnostic plot

if plot_diagnostic:
    pbottom = np.floor(dpkit.search_extreme_xarray(p[0, :, :], mode='min', limsh=True, cntLat=inilat, cntLon=inilon, radius=tccShRadius)[0])

    #
    width = 15.8
    height = 10
    figTerm1 = plt.figure(figsize=(width, 10))
    axTerm1 = figTerm1.add_axes([0.2/width, 0.67, 6/width, 3/10])
    axTerm2 = figTerm1.add_axes([0.2/width, 0.34, 6/width, 3/10])
    axTerm3 = figTerm1.add_axes([0.2/width, 0.01, 6/width, 3/10])

    axZbottom = figTerm1.add_axes([6.4/width, 0.01, 3/width, 3/10], projection=cart_proj)
    axWindbottom = figTerm1.add_axes([9.6/width, 0.01, 3/width, 3/10], projection=cart_proj)
    axZ850p = figTerm1.add_axes([6.4/width, 0.34, 3/width, 3/10], projection=cart_proj)
    axWind850p = figTerm1.add_axes([9.6/width, 0.34, 3/width, 3/10], projection=cart_proj)
    axZ700p = figTerm1.add_axes([6.4/width, 0.67, 3/width, 3/10], projection=cart_proj)
    axWind700p = figTerm1.add_axes([9.6/width, 0.67, 3/width, 3/10], projection=cart_proj)

    plot_force_radiusHeight(axTerm1, term1)
    plot_force_radiusHeight(axTerm2, term2)
    plot_force_radiusHeight(axTerm3, term3)

    u = getvar(ncfile, 'ua')
    v = getvar(ncfile, 'va')
    ubottom = vinterp(ncfile, u, 'pressure', [pbottom])[0, :, :]
    vbottom = vinterp(ncfile, v, 'pressure', [pbottom])[0, :, :]
    u850 = vinterp(ncfile, u, 'pressure', [850])[0, :, :]
    v850 = vinterp(ncfile, v, 'pressure', [850])[0, :, :]
    u700 = vinterp(ncfile, u, 'pressure', [700])[0, :, :]
    v700 = vinterp(ncfile, v, 'pressure', [700])[0, :, :]
    z = getvar(ncfile, 'height')
    zbottom = vinterp(ncfile, z, 'pressure', [pbottom])
    z850 = vinterp(ncfile, z, 'pressure', [850])
    z700 = vinterp(ncfile, z, 'pressure', [700])

    plt_tand_rad_wind(axWindbottom, ubottom, vbottom, minusMean=False)
    plt_tand_rad_wind(axWind850p, u850, v850, minusMean=False)
    plt_tand_rad_wind(axWind700p, u700, v700, minusMean=False)

    plot_z(axZbottom, zbottom)
    plot_z(axZ850p, z850)
    plot_z(axZ700p, z700)

    axTerm1.set_title("pressure gradiant")
    axTerm2.set_title("centrifugal force")
    axTerm3.set_title("Coriolis force")
    axZbottom.set_title(str(int(pbottom))+"hpa height diff")
    axZ850p.set_title("850hpa height diff")
    axZ700p.set_title("700hpa height diff")
    axWindbottom.set_title(str(int(pbottom))+"hpa windfield")
    axWind850p.set_title("850hpa windfield")
    axWind700p.set_title("700hpa windfield")

    axTerm1.set_xticklabels("")
    axTerm2.set_xticklabels("")

# %%
figNRF = plt.figure(figsize=(12, 9))
axNRF = figNRF.add_axes([0.1, 0.1, 0.8, 0.8])
plot_force_radiusHeight(axNRF, term1+term2+term3)
figNRF.savefig(figName, dpi=200)
