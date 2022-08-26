# %%
import datetime
import numpy as np
import pandas as pd

reference = "https://www.jma.go.jp/jma/jma-eng/jma-center/rsmc-hp-pub-eg/Besttracks/e_format_bst.html"
headerDescription = {'internationalID': 'International number ID,  Last two digits of calendar year followed by 2-digit serial number ID of the storm of Tropical Storm(TS) intensity or greater',
                     'datalines': 'Number of data lines',
                     'tcID': 'Tropical cyclone number ID, Serial number ID of the storm of intensity with maximum sustained wind speed of 28 kt (near gale) or  greater',
                     'outjma': 'Flag of the last data line,  0 : Dissipation, 1 : Going out of the responsible area of Japan Meteorological',
                     'diffana': 'Difference between the time of the last data and the time of the final analysis (Unit : hour)',
                     'name': 'Name of the storm',
                     'revDate': 'Date of the latest revision'}
dataDescription = {'time': 'Time of analysis (UTC)',
                   'grade': 'Grade',
                   'lat': 'Latitude of the center. Unit : degree',
                   'lon': 'Longitude of the center. Unit : degree',
                   'cntP': 'Central pressure. Unit : hPa',
                   'maxws': 'Maximum sustained wind speed. Unit : knot (kt)',
                   'dtLg50kt': 'Direction of the longest radius of 50kt winds or greater',
                   'rdLg50kt': 'The longest radius of 50kt winds or greater. Unit : nautical mile (nm)',
                   'rdSt50kt': 'The shortest radius of 50kt winds or greater. Unit : nautical mile (nm)',
                   'dtLg30kt': 'Direction of the longest radius of 30kt winds or greater',
                   'rdLg30kt': 'The longest radius of 30kt winds or greater. Unit : nautical mile (nm)',
                   'rdSt30kt': 'The shortest radius of 30kt winds or greater. Unit : nautical mile (nm)',
                   'landfallJP': 'Indicator of landfall or passage, Landfall or passage over the Japanese islands occurred within one hour after the time of the analysis with this indicator'}


def parse_header(headerStr: str):
    indicator = headerStr[0:5]
    if indicator != "66666":
        print("wrong indicator: "+indicator)

    internationalID = headerStr[6:10]
    datalines = int(headerStr[12:15])

    tcID = headerStr[16:20]
    internationalID2 = headerStr[21:25]

    if internationalID2 != internationalID:
        print("not equal internationalID?? : "+str(internationalID)+" and "+str(internationalID2))

    outjma = bool(int(headerStr[26]))
    diffana = int(headerStr[28])
    name = headerStr[30:50].strip()
    revDate = datetime.datetime.strptime(headerStr[59:72].strip(), '%Y%m%d')

    header = {'internationalID': internationalID, 'datalines': datalines,
              'tcID': tcID, 'outjma': outjma, 'diffana': diffana,
              'name': name, 'revDate': revDate}
    return header


def parse_data(dataStr: str):
    if int(dataStr[0:2]) > 50:
        time = datetime.datetime.strptime('19'+dataStr[0:8], '%Y%m%d%H')
    else:
        time = datetime.datetime.strptime('20'+dataStr[0:8], '%Y%m%d%H')

    indicator = dataStr[9:12]
    if indicator != "002":
        print("wrong indicator: "+indicator)

    grade = str2grade(dataStr[13])
    lat = emptyStr2Float(dataStr[15:18])/10
    lon = emptyStr2Float(dataStr[19:23])/10
    cntP = emptyStr2Float(dataStr[24:28])
    maxws = emptyStr2Float(dataStr[33:36])

    dtLg50kt = str2direct(dataStr[41])
    rdLg50kt = emptyStr2Float(dataStr[42:46])

    rdSt50kt = emptyStr2Float(dataStr[47:51])
    dtLg30kt = str2direct(dataStr[52])

    rdLg30kt = emptyStr2Float(dataStr[53:57])
    rdSt30kt = emptyStr2Float(dataStr[58:62])

    landfallJP = dataStr[71]
    if landfallJP == "#":
        landfallJP = True
    else:
        landfallJP = False

    data = {'time': time, 'grade': grade,
            'lat': lat, 'lon': lon,
            'cntP': cntP, 'maxws': maxws,
            'dtLg50kt': dtLg50kt, 'rdLg50kt': rdLg50kt, 'rdSt50kt': rdSt50kt,
            'dtLg30kt': dtLg30kt, 'rdLg30kt': rdLg30kt, 'rdSt30kt': rdSt30kt,
            'landfallJP': landfallJP}
    return data


def emptyStr2Float(inStr: str):
    if inStr.strip() == '':
        outFloat = 0.0
    else:
        outFloat = float(inStr)
    return outFloat


def str2grade(inStr: str):
    gradeStr = inStr
    if inStr == "2":
        gradeStr = "Tropical Depression"
    elif inStr == "3":
        gradeStr = "Tropical Storm"
    elif inStr == "4":
        gradeStr = "Severe Tropical Storm"
    elif inStr == "5":
        gradeStr = "Typhoon"
    elif inStr == "6":
        gradeStr = "Extra-tropical Cyclone"
    elif inStr == "7":
        gradeStr = "Just entering into the responsible area of Japan Meteorological Agency"
    elif inStr == "9":
        gradeStr = "Tropical Cyclone of TS intensity or higher"
    return gradeStr


def str2direct(inStr: str):
    directStr = inStr
    if inStr == "1":
        directStr = "Northeast"
    elif inStr == "2":
        directStr = "East"
    elif inStr == "3":
        directStr = "Southeast"
    elif inStr == "4":
        directStr = "South"
    elif inStr == "5":
        directStr = "Southwest"
    elif inStr == "6":
        directStr = "West"
    elif inStr == "7":
        directStr = "Northwest"
    elif inStr == "8":
        directStr = "North"
    elif inStr == "9":
        directStr = "symmetric"
    return directStr


def read_a_tc_dataset(infileh):
    if type(infileh) is str:
        infileh = open(infileh)
    headerStr = infileh.readline()
    header = parse_header(headerStr)
    datalines = header['datalines']

    time = []
    grade = []
    lat = np.zeros(datalines)
    lon = np.zeros(datalines)
    cntP = np.zeros(datalines)
    maxws = np.zeros(datalines)
    dtLg50kt = []
    rdLg50kt = np.zeros(datalines)
    rdSt50kt = np.zeros(datalines)
    dtLg30kt = []
    rdLg30kt = np.zeros(datalines)
    rdSt30kt = np.zeros(datalines)
    landfallJP = []

    for idata in range(0, datalines):
        dataStr = infileh.readline()
        dataline = parse_data(dataStr)
        time.append(dataline['time'])
        grade.append(dataline['grade'])
        lat[idata] = dataline['lat']
        lon[idata] = dataline['lon']
        cntP[idata] = dataline['cntP']
        maxws[idata] = dataline['maxws']
        dtLg50kt.append(dataline['dtLg50kt'])
        rdLg50kt[idata] = dataline['rdLg50kt']
        rdSt50kt[idata] = dataline['rdSt50kt']
        dtLg30kt.append(dataline['dtLg30kt'])
        rdLg30kt[idata] = dataline['rdLg30kt']
        rdSt30kt[idata] = dataline['rdSt30kt']
        landfallJP.append(dataline['landfallJP'])

    data = pd.DataFrame({'time': time, 'grade': grade,
                         'lat': lat, 'lon': lon,
                         'cntP': cntP, 'maxws': maxws,
                         'dtLg50kt': dtLg50kt, 'rdLg50kt': rdLg50kt, 'rdSt50kt': rdSt50kt,
                         'dtLg30kt': dtLg30kt, 'rdLg30kt': rdLg30kt, 'rdSt30kt': rdSt30kt,
                         'landfallJP': landfallJP})
    return header, data
