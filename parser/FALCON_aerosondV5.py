# %%
import pandas as pd
import numpy as np
from woad import diagkit, dpkit

column_names = ['type', 'deviceCode', 'millstr1', 'null1',
                'pressure', 'temp', 'RH', 'PM10', 'PM25', 'PM100',
                'lat', 'lon', 'angle', 'speed',
                'SNR', 'RSSI', 'date', 'time', 'seq', 'null2', 'millstr2', 'null3']

# %%
UNIV_GAS_CONSTANT = 8.31432
EARCH_GRAVITY_CONSTANT = 9.80665
MOLAR_MASS_OF_AIR = 0.0289644
TEMP_LAPS_RATE = -0.0065

def pressure_to_height1(pressure, temperature, baseHeight=0, basePressure=None):
    if basePressure is None:
        basePressure = pressure[0]

    RLgM = -(UNIV_GAS_CONSTANT * (temperature + 273.15)) / EARCH_GRAVITY_CONSTANT * MOLAR_MASS_OF_AIR
    height = (baseHeight + RLgM * np.log(pressure / basePressure))
    return height*1000

def pressure_to_height2(pressure, temperature, seaLevelPressure=101325.0):
    RLgM = UNIV_GAS_CONSTANT * TEMP_LAPS_RATE / EARCH_GRAVITY_CONSTANT * MOLAR_MASS_OF_AIR
    p1 = (pressure / seaLevelPressure) ** RLgM
    t0 = temperature + 273.15
    height = t0 * (1.0 - p1) / TEMP_LAPS_RATE
    return height*1000

def height_to_pressure(height, temperature, basePressure, baseHeight=0):
    RLgM = -(UNIV_GAS_CONSTANT * (temperature + 273.15)) / EARCH_GRAVITY_CONSTANT * MOLAR_MASS_OF_AIR
    pressure = basePressure * np.exp((height-baseHeight)/1000/RLgM)
    return pressure

def find_startIdx(timeList):
    for startIdx in range(timeList.size):
        if timeList[startIdx][3] == '0':
            break
    return startIdx

def find_invalidMoveIdx(distance, speed,
                        distThreshold=100*1000, distDiffThreshold=50, spdThreshold=100, spdDiffThreshold=25):
    distDiff = np.abs(distance[1:]-distance[0:-1])
    speedDiff = np.abs(speed[1:]-speed[0:-1])
    invalidMoveIdx_dist = np.where(distance>distThreshold)[0]
    invalidMoveIdx_distDiff = np.where(distDiff>distDiffThreshold)[0]+1
    invalidMoveIdx_speed = np.where(speed>spdThreshold)[0]
    invalidMoveIdx_spdDiff = np.where(speedDiff>spdDiffThreshold)[0]+1
    invalidMoveIdx = np.unique(np.concatenate([invalidMoveIdx_dist,
                                               invalidMoveIdx_distDiff,
                                               invalidMoveIdx_speed,
                                               invalidMoveIdx_spdDiff]))
    return invalidMoveIdx

# %%
# ENVR, aa000123, 1677520105863, 0, 102113.0, 14.44, 85.56,   36,   58,    74, 23.8739904, 120.5814144,   0.0,   0.0, -55.0, 12.0, 23/2/28, 01:48:25, 134, 0, 1677520105863,
#     ,   serial,           sec, 0, pressure,  temp,    RH, PM10, PM25, PM100,        lat,         lon, angle, speed,   snr, rssi,    date,     time, seq, 0,           sec,

# private String[] columnNames = new String[] { "編號", "運作時間", "序號", "溫度", "溼度", "氣壓", "PM10", "PM25", "PM100", "緯度", "經度", "衛星時間", "接收時間" };

def read(filepath):
    dataRaw = pd.read_csv(filepath, names=column_names, skiprows=4)

    startIdx = find_startIdx(dataRaw['time'])

    pressure = (np.array(dataRaw['pressure'])/100)[startIdx:]
    temp = np.array(dataRaw['temp'])[startIdx:]
    RH = np.array(dataRaw['RH'])[startIdx:]
    PM10 = np.array(dataRaw['PM10'])[startIdx:]
    PM25 = np.array(dataRaw['PM25'])[startIdx:]
    PM100 = np.array(dataRaw['PM100'])[startIdx:]
    lat = np.array(dataRaw['lat'])[startIdx:]
    lon = np.array(dataRaw['lon'])[startIdx:]
    angle = np.array(dataRaw['angle'])[startIdx:]
    speed = np.array(dataRaw['speed'])[startIdx:]
    time = dataRaw['time'][startIdx:]

    height = pressure_to_height1(pressure*100, temp)
    pottemp = diagkit.temp_to_pottemp(temp_inDegK=temp+273.15,
                                      pressure_inhPa=pressure)
    specificHumidity = diagkit.RH_to_specificHumidity(RH_inPercent=RH/100,
                                                      temp_inDegC=temp,
                                                      pressure_inhPa=pressure)
    dewPoint = diagkit.RH_to_dewPointTemp(RH, temp)

    dist = dpkit.cal_distance(lats=lat, lons=lon, cntLat=lat[0], cntLon=lon[0])
    invalidMoveIdx = find_invalidMoveIdx(dist, speed)
    angle[invalidMoveIdx] = np.nan
    speed[invalidMoveIdx] = np.nan
    dist[invalidMoveIdx] = np.nan

    return {'pressure': pressure, 'temp': temp, 'RH': RH,
            'PM10': PM10, 'PM25': PM25, 'PM100': PM100,
            'lat': lat, 'lon': lon, 'angle': angle, 'speed': speed, 'time': time,
            'height': height, 'pottemp': pottemp, 'specificHumidity': specificHumidity,
            'dewPoint':dewPoint,
            'dist': dist, 'invalidMoveIdx': invalidMoveIdx, 'startIdx': startIdx,
            'dataRaw': dataRaw}
