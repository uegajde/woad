# %%
import numpy as np
import pandas as pd
from datetime import datetime
from woad import parameter as parm

column_names = ["BASIN", "CY", "Time", "TECHNUM", "TECH", "TAU", "Lat", "Lon", "VMAX", "MSLP",
                "TY", "RAD", "WINDCODE", "RAD1", "RAD2", "RAD3", "RAD4", "RADP", "RRP", "MRD",
                "GUSTS", "EYE", "SUBREGION", "MAXSEAS", "INITIALS", "DIR", "SPEED", "STORMNAME", "DEPTH", "SEAS", "SEASCODE"]
# column name: https://www.usno.navy.mil/NOOC/nmfc-ph/RSS/jtwc/best_tracks/wpindex.php


def read(JTWC_BestTrackFile, parse_Lat=True, parse_Lon=True, parse_Time=True, knot2ms_VMAX=True):

    data = pd.read_csv(JTWC_BestTrackFile, names=column_names)

    if parse_Lat:
        for ilat in range(0, data['Lat'].size):
            latStr = data['Lat'][ilat]
            lonNum = np.float(latStr[:-1])/10.0

            if latStr[-1] == "N":
                data['Lat'][ilat] = lonNum
            elif latStr[-1] == "S":
                data['Lat'][ilat] = lonNum*-1

    if parse_Lon:
        for ilon in range(0, data['Lon'].size):
            lonStr = data['Lon'][ilon]
            lonNum = np.float(lonStr[:-1])/10.0

            if lonStr[-1] == "E":
                data['Lon'][ilon] = lonNum
            elif lonStr[-1] == "W":
                data['Lon'][ilon] = lonNum*-1

    if parse_Time:
        for itime in range(0, data['Time'].size):
            data['Time'][itime] = datetime.strptime(str(data['Time'][itime]), "%Y%m%d%H")

    if knot2ms_VMAX:
        for iVal in range(0, data['VMAX'].size):
            data['VMAX'][iVal] = data['VMAX'][iVal]*parm.knot2ms

    return data
