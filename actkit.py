from wrf import interpline, CoordPair, to_np, vertcross, latlon_coords
from scipy.interpolate import interp1d
import numpy as np
import xarray
from cartopy.geodesic import Geodesic
from woad import parameter as parm

cgeo = Geodesic(radius=parm.earth_radius)


def search_extreme_xarray(var: xarray.DataArray, mode: str, limsh=False, cntLat=None, cntLon=None, radius=2):
    if limsh:
        if (cntLat is None) or (cntLon is None):
            print('Did not get initial guess of "cntLat" and "cntLon", limsh is truned off.')
            limsh = False

    if limsh:
        var = var.where(abs(var.coords['XLAT']-cntLat) < radius, drop=True).squeeze()
        var = var.where(abs(var.coords['XLONG']-cntLon) < radius, drop=True).squeeze()

    if mode == 'max':
        exValue = var.max()
    elif mode == 'min':
        exValue = var.min()
    else:
        print('diagkit.searchExtreme: not got valid mode, uses max by default')
        exValue = var.max()

    exLat = var.coords['XLAT'].where(var == exValue, drop=True).squeeze()
    exLon = var.coords['XLONG'].where(var == exValue, drop=True).squeeze()

    return to_np(exValue), to_np(exLat), to_np(exLon)


def interp_to_polarCoord2D_xarray(var: xarray.DataArray, ncfile, cntLat, cntLon, angles, radiuses):
    varInPCS = np.empty([len(angles), len(radiuses)])
    latInPCS = np.empty([len(angles), len(radiuses)])
    lonInPCS = np.empty([len(angles), len(radiuses)])

    start_point = CoordPair(lat=cntLat, lon=cntLon)
    end_points = cgeo.direct([cntLon, cntLat], angles, radiuses[-1]+10000)

    lats, lons = latlon_coords(var)
    lats, lons = to_np(lats), to_np(lons)
    cntVar = interp_2d_to_point_lonlat_3plinear(to_np(var), lonMap=to_np(lons), latMap=to_np(lats), lonPoint=cntLon, latPoint=cntLat)

    for idx in range(len(angles)):
        end_point = CoordPair(lat=end_points[idx, 1], lon=end_points[idx, 0])

        temp = interpline(var, wrfin=ncfile, start_point=start_point, end_point=end_point, latlon=True)

        lons, lats = collect_1DLatLon_in_xyloc_xarray(temp)
        temp = to_np(temp)

        if lons[0] != cntLon:
            lons = np.append(cntLon, lons)
            lats = np.append(cntLat, lats)
            temp = np.append(cntVar, temp)

        dist = cal_distance(lats, lons, cntLat, cntLon)

        intpfuncVar = interp1d(dist, temp)
        intpfuncLat = interp1d(dist, lats)
        intpfuncLon = interp1d(dist, lons)

        varInPCS[idx, ] = intpfuncVar(radiuses)
        latInPCS[idx, ] = intpfuncLat(radiuses)
        lonInPCS[idx, ] = intpfuncLon(radiuses)

    return varInPCS, latInPCS, lonInPCS


def interp_to_polarCoord3D_xarray(var: xarray.DataArray, zVar, ncfile, cntLat, cntLon, angles, radiuses, zlevels):
    varInPCS = np.empty([len(zlevels), len(angles), len(radiuses)])
    latInPCS = np.empty([len(angles), len(radiuses)])
    lonInPCS = np.empty([len(angles), len(radiuses)])

    start_point = CoordPair(lat=cntLat, lon=cntLon)
    end_points = cgeo.direct([cntLon, cntLat], angles, radiuses[-1]+10000)

    lats, lons = latlon_coords(var)
    lats, lons = to_np(lats), to_np(lons)
    cntVarTemp = np.empty(zVar.shape[0])
    cntZVarTemp = np.empty(zVar.shape[0])
    for ilev in range(zVar.shape[0]):
        cntVarTemp[ilev] = interp_2d_to_point_lonlat_3plinear(to_np(var[ilev, :, :]), lonMap=to_np(lons), latMap=to_np(lats), lonPoint=cntLon, latPoint=cntLat)
        cntZVarTemp[ilev] = interp_2d_to_point_lonlat_3plinear(to_np(zVar[ilev, :, :]), lonMap=to_np(lons), latMap=to_np(lats), lonPoint=cntLon, latPoint=cntLat)
    intpfunc = interp1d(cntZVarTemp, cntVarTemp, bounds_error=False)
    cntVar = intpfunc(zlevels)

    for iangle in range(len(angles)):
        end_point = CoordPair(lat=end_points[iangle, 1], lon=end_points[iangle, 0])

        temp = vertcross(var, vert=zVar, levels=zlevels, wrfin=ncfile, start_point=start_point, end_point=end_point, latlon=True)
        lons, lats = collect_1DLatLon_in_xyloc_xarray(temp)

        if lons[0] != cntLon:
            lons = np.append(cntLon, lons)
            lats = np.append(cntLat, lats)
            temp = np.append(np.reshape(cntVar, [cntVar.size, 1]), temp, axis=1)

        dist = cal_distance(lats, lons, cntLat, cntLon)

        intpfuncLat = interp1d(dist, to_np(lats))
        intpfuncLon = interp1d(dist, to_np(lons))
        latInPCS[iangle, ] = intpfuncLat(radiuses)
        lonInPCS[iangle, ] = intpfuncLon(radiuses)

        temp = to_np(temp)

        for ilevel in range(len(zlevels)):
            intpfuncVar = interp1d(dist, temp[ilevel])
            varInPCS[ilevel, iangle, ] = intpfuncVar(radiuses)
    return varInPCS, latInPCS, lonInPCS


def collect_1DLatLon_in_xyloc_xarray(xVar: xarray.DataArray):
    lats = np.empty([len(xVar.xy_loc)])
    lons = np.empty([len(xVar.xy_loc)])
    temp = to_np(xVar.xy_loc)
    for icoord in range(len(xVar.xy_loc)):
        lats[icoord] = temp[icoord].lat
        lons[icoord] = temp[icoord].lon
    return lons, lats


def cal_distance(lats, lons, cntLat, cntLon):
    lonlat = np.concatenate((np.reshape(lons, [lons.size, 1]), np.reshape(lats, [lats.size, 1])), axis=1)
    dist = np.array(cgeo.inverse([cntLon, cntLat], lonlat)[:, 0])
    dist = np.reshape(dist, lons.shape)
    return dist


def smooth_2DArray(dataIn, cntWeight=2, rndWeight=1, loop=1):
    if not len(dataIn.shape) == 2:
        print('warnning: not a 2D array, return the raw data')
        return dataIn

    temp = np.empty(dataIn.shape)
    temp[:] = dataIn[:]
    dataOut = np.empty(dataIn.shape)

    for iter in range(loop):
        dataOut[1: -1, 1: -1] = (temp[1: -1, 1: -1]*cntWeight +
                                 (temp[: -2, 1: -1] +
                                  temp[2:, 1: -1] +
                                  temp[1: -1, : -2] +
                                  temp[1: -1, 2:])*rndWeight)/(cntWeight+rndWeight*4)
        dataOut[0, 1: -1] = (temp[0, 1: -1]*cntWeight +
                             (temp[0, 0: -2]+temp[0, 2:]+temp[1, 1: -1])*rndWeight)/(cntWeight+rndWeight*3)
        dataOut[-1, 1: -1] = (temp[-1, 1: -1]*cntWeight +
                              (temp[-1, 0: -2]+temp[-1, 2:]+temp[-2, 1: -1])*rndWeight)/(cntWeight+rndWeight*3)
        dataOut[1: -1, 0] = (temp[1: -1, 0]*cntWeight +
                             (temp[0: -2, 0]+temp[2:, 0]+temp[1: -1, 1])*rndWeight)/(cntWeight+rndWeight*3)
        dataOut[1: -1, -1] = (temp[1: -1, -1]*cntWeight +
                              (temp[0: -2, -1]+temp[2:, -1]+temp[1: -1, -2])*rndWeight)/(cntWeight+rndWeight*3)
        dataOut[0, 0] = (temp[0, 0]*cntWeight+(temp[0, 1]+temp[1, 0])*rndWeight)/(cntWeight+rndWeight*2)
        dataOut[0, -1] = (temp[0, -1]*cntWeight+(temp[0, -2]+temp[1, -1])*rndWeight)/(cntWeight+rndWeight*2)
        dataOut[-1, 0] = (temp[-1, 0]*cntWeight+(temp[-1, 1]+temp[-2, 0])*rndWeight)/(cntWeight+rndWeight*2)
        dataOut[-1, -1] = (temp[-1, -1]*cntWeight+(temp[-1, -2]+temp[-2, -1])*rndWeight)/(cntWeight+rndWeight*2)
        temp[:] = dataOut[:]

    return dataOut


def interp_2d_to_point_lonlat_3plinear(varMain, lonMap, latMap, lonPoint, latPoint):
    # intepolate to a specific point (lon,lat) with 3 point linear interpolation

    varMain = np.reshape(varMain, [varMain.size])
    lon = np.reshape(lonMap, [lonMap.size, 1])
    lat = np.reshape(latMap, [latMap.size, 1])
    lonlat = np.concatenate((lon, lat), axis=1)
    dist = np.array(cgeo.inverse([lonPoint, latPoint], lonlat)[:, 0])

    rankDist = np.argsort(dist)

    var3 = varMain[rankDist[0:3]]
    x = (lon[rankDist[0:3]]-lonPoint)*100
    y = (lat[rankDist[0:3]]-latPoint)*100

    A1 = x[1]*y[2]-x[2]*y[1]
    A2 = x[2]*y[0]-x[0]*y[2]
    A3 = x[0]*y[1]-x[1]*y[0]

    varInterp = (var3[0]*A1+var3[1]*A2+var3[2]*A3)/(A1+A2+A3)

    return varInterp[0]


def cal_trimed_lonlat_idx(lon, lat, trimLonLat):
    # calculate the index covering the specified trimLonLat
    # an universal version of trim_data_xarray
    # trimLonLat = [minLon, maxLon, minLat, maxLat]
    idxMinX = 0
    idxMinY = 0
    idxMaxY, idxMaxX = lon.shape

    for iter in np.arange(4):
        idxMinYLoc = np.where(np.max(lat, axis=1) < (trimLonLat[2]))[0][-1]
        idxMaxYLoc = np.where(np.min(lat, axis=1) > (trimLonLat[3]))[0][0]
        idxMinXLoc = np.where(np.max(lon, axis=0) < (trimLonLat[0]))[0][-1]
        idxMaxXLoc = np.where(np.min(lon, axis=0) > (trimLonLat[1]))[0][0]

        lat = lat[idxMinYLoc:idxMaxYLoc+1, idxMinXLoc:idxMaxXLoc+1]
        lon = lon[idxMinYLoc:idxMaxYLoc+1, idxMinXLoc:idxMaxXLoc+1]

        idxMaxX = idxMinX+idxMaxXLoc+1
        idxMaxY = idxMinY+idxMaxYLoc+1
        idxMinX = idxMinX+idxMinXLoc
        idxMinY = idxMinY+idxMinYLoc

    return idxMinX, idxMaxX, idxMinY, idxMaxY


def trim_data_xarray(data, trimLonLat):
    # trim data to only cover the specified trimLonLat
    # trimLonLat = [minLon, maxLon, minLat, maxLat]
    data = data.where(data.coords['XLAT'] > trimLonLat[2], drop=True).squeeze()
    data = data.where(data.coords['XLAT'] < trimLonLat[3], drop=True).squeeze()
    data = data.where(data.coords['XLONG'] > trimLonLat[0], drop=True).squeeze()
    data = data.where(data.coords['XLONG'] < trimLonLat[1], drop=True).squeeze()
    return data
