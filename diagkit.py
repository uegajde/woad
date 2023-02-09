from wrf import getvar, to_np
import numpy as np
from woad import parameter as parm
from woad import actkit


def uv10(ncfile, roughness=0.0001, mask_height=20):
    # zhmkz1 = h*(1-h/earth_radius)
    # u10 = u*(log10(10/0.0001)/log10(zhmkz1/0.0001))

    u = getvar(ncfile, 'ua', units="m s-1")[0, :, :]
    v = getvar(ncfile, 'va', units="m s-1")[0, :, :]
    h = getvar(ncfile, 'height', units="m")[0, :, :]
    terheight = getvar(ncfile, 'ter', units='m')

    factor = (np.log10(10/roughness)/np.log10(h*(1-h/parm.earth_radius)/roughness))

    u10 = u*factor
    v10 = v*factor

    if mask_height >= 0:
        u10 = u10.where(terheight < mask_height)
        v10 = v10.where(terheight < mask_height)

    return u10, v10


def tc_wind_InPCS(ncfile, cntLat: float, cntLon: float, wspd=None,
                  angleInterval=5, radiusesEnd=400000, radiusesInterval=1000):
    """
        rmw, r34, maxmeanws, midproduct = diagkit.tc_wind_InPCS()

        input:
        ncfile: ncfile opened with netcdf4.Dataset().
        wspd: windspeed data taken from the wrf-py.
        cntLat: lat of tc center.
        cntLon: lon of tc center.
        angleInterval: the interval to interpolate by angle.
        radiusesEnd: the distance to be interpolated.
        radiusesInterval: the interval to interpolate by distance.

        output:
        rmw: radius of maximun windspeed
        r34: radius of 34-knots
        maxmeanws: maximun mean windspeed
        midproduct['angles']: the angle constructing cylindrical coordinate
        midproduct['radiuses']: the ridus constructing cylindrical coordinate
        midproduct['wsInPCS']: windspeed at cylindrical coordinate
        midproduct['latInPCS']: lat at cylindrical coordinate
        midproduct['lonInPCS']: lon at cylindrical coordinate
    """

    # process args
    if wspd is None:
        u10, v10 = uv10(ncfile)
        wspd = (u10**2+v10**2)**0.5

    angles = np.arange(0, 360, angleInterval)
    radiuses = np.arange(0, radiusesEnd+radiusesInterval, radiusesInterval)

    wsInPCS, latInPCS, lonInPCS = actkit.interp_to_polarCoord2D_xarray(wspd, ncfile, cntLat, cntLon, angles, radiuses)

    wsInPCSRMean = np.nanmean(wsInPCS, axis=0)
    maxmeanws = np.nanmax(wsInPCSRMean)
    rmw = radiuses[wsInPCSRMean == maxmeanws][0]

    if np.nanmax(to_np(wsInPCSRMean)) > 34*parm.knot2ms:
        r34 = np.nanmax(radiuses[wsInPCSRMean > 34*parm.knot2ms])
    else:
        r34 = -99.9

    midproduct = {}
    midproduct['wsInPCS'] = wsInPCS
    midproduct['angles'] = angles
    midproduct['radiuses'] = radiuses
    midproduct['latInPCS'] = latInPCS
    midproduct['lonInPCS'] = lonInPCS

    return rmw, r34, maxmeanws, midproduct


def tanRadWind_InPCS(uInPCS, vInPCS, angles):
    tanWind = np.empty(uInPCS.shape)
    radWind = np.empty(uInPCS.shape)

    if len(uInPCS.shape) == 2:
        for iangle in range(len(angles)):
            rad = np.deg2rad(angles[iangle])
            tanWind[iangle, ] = -np.cos(rad)*uInPCS[iangle, ]+np.sin(rad)*vInPCS[iangle, ]
            radWind[iangle, ] = np.sin(rad)*uInPCS[iangle, ]+np.cos(rad)*vInPCS[iangle, ]
        tanWind[:, 0] = 0
        radWind[:, 0] = 0
    elif len(uInPCS.shape) == 3:
        for iangle in range(len(angles)):
            rad = np.deg2rad(angles[iangle])
            tanWind[:, iangle, ] = -np.cos(rad)*uInPCS[:, iangle, ]+np.sin(rad)*vInPCS[:, iangle, ]
            radWind[:, iangle, ] = np.sin(rad)*uInPCS[:, iangle, ]+np.cos(rad)*vInPCS[:, iangle, ]
        tanWind[:, :, 0] = 0
        radWind[:, :, 0] = 0

    return tanWind, radWind


def netRadialForce_pCoord(ncfile, u, v, p, z, cntLat, cntLon, plevels, angleInterval=5, radiusesEnd=350000, radiusesInterval=2000):
    g = parm.gravitational_acceleration(cntLat)
    f = parm.coriolis_coefficient(cntLat)
    angles = np.arange(0, 360, angleInterval)
    radiuses = np.arange(0, radiusesEnd+radiusesInterval, radiusesInterval)
    radiusesForZ = np.arange(0, radiusesEnd+radiusesInterval*2, radiusesInterval)

    uInPCS, _, _ = actkit.interp_to_polarCoord3D_xarray(var=u, zVar=p, ncfile=ncfile,
                                                        cntLat=cntLat, cntLon=cntLon,
                                                        angles=angles, radiuses=radiuses, zlevels=plevels)
    vInPCS, _, _ = actkit.interp_to_polarCoord3D_xarray(var=v, zVar=p, ncfile=ncfile,
                                                        cntLat=cntLat, cntLon=cntLon,
                                                        angles=angles, radiuses=radiuses, zlevels=plevels)
    zInPCS, _, _ = actkit.interp_to_polarCoord3D_xarray(var=z, zVar=p, ncfile=ncfile,
                                                        cntLat=cntLat, cntLon=cntLon,
                                                        angles=angles, radiuses=radiusesForZ, zlevels=plevels)

    tanWind, _ = tanRadWind_InPCS(uInPCS, vInPCS, angles)
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

    midproduct = {}
    midproduct['uInPCS'] = uInPCS
    midproduct['vInPCS'] = vInPCS
    midproduct['zInPCS'] = zInPCS
    return netRadialForce, midproduct


# def netRadialForce_zCoord(ncfile, u, v, p, z, cntLat, cntLon, zlevels, angleInterval=5, radiusesEnd=350000, radiusesInterval=2000):
#     f = coriolis_coefficient(cntLat)
#     # rho =
#     angles = np.arange(0, 360, angleInterval)
#     radiuses = np.arange(0, radiusesEnd+radiusesInterval, radiusesInterval)
#     radiusesForZ = np.arange(0, radiusesEnd+radiusesInterval*2, radiusesInterval)

#     uInPCS, _, _ = interp_to_polarCoord3D_xarray(var=u, zVar=z, ncfile=ncfile,
#                                         cntLat=cntLat, cntLon=cntLon,
#                                         angles=angles, radiuses=radiuses, zlevels=zlevels)
#     vInPCS, _, _ = interp_to_polarCoord3D_xarray(var=v, zVar=z, ncfile=ncfile,
#                                         cntLat=cntLat, cntLon=cntLon,
#                                         angles=angles, radiuses=radiuses, zlevels=zlevels)
#     pInPCS, _, _ = interp_to_polarCoord3D_xarray(var=p, zVar=z, ncfile=ncfile,
#                                         cntLat=cntLat, cntLon=cntLon,
#                                         angles=angles, radiuses=radiusesForZ, zlevels=zlevels)

#     tanWind, _ = tanRadWind_InPCS(uInPCS, vInPCS, angles)
#     tanWindAzMean = np.mean(tanWind, 1)
#     pAzMean = np.mean(pInPCS, 1)

#     term1 = np.zeros(tanWindAzMean.shape)
#     term1temp = -rho*(pAzMean[:, 2:]-pAzMean[:, :-2])/2
#     term1[:, 1:] = term1temp

#     term2 = np.empty(tanWindAzMean.shape)
#     term2 = tanWindAzMean**2/radiuses
#     term2[:, 0] = 0

#     term3 = f*tanWindAzMean

#     netRadialForce = term1+term2+term3
#     return netRadialForce


def tcc_by_pressureCentroid(p: np.ndarray, lon: np.ndarray, lat: np.ndarray, initLon, initLat, bgRad=500000, shRad=100000, maxIter=20):
    p = np.reshape(p, [p.size, 1])
    lon = np.reshape(lon, [lon.size, 1])
    lat = np.reshape(lat, [lat.size, 1])
    newLon, newLat = initLon, initLat

    for iter in range(maxIter):
        dist = actkit.cal_distance(lat, lon, initLat, initLon)
        penv = np.mean(p[dist <= bgRad])
        pdeficit = penv-p

        maskSH = dist <= shRad
        newLon = np.sum(lon[maskSH]*pdeficit[maskSH])/np.sum(pdeficit[maskSH])
        newLat = np.sum(lat[maskSH]*pdeficit[maskSH])/np.sum(pdeficit[maskSH])

        if newLon == initLon and newLat == initLat:
            break
        else:
            initLon = newLon
            initLat = newLat

            if iter == maxIter-1:
                print('info: tcc locating result not coverge perfectly for current maxIter')

    return newLon, newLat
