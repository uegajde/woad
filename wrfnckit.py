import numpy as np
from wrf import interplevel

def collectEnsemble(ncList, varName):
    ensLen = len(ncList)
    xSize = np.array(ncList[0][varName]).shape[1:]
    if len(xSize) == 2:
        xEnsFull = np.zeros((ensLen, xSize[0], xSize[1]))
    elif len(xSize) == 3:
        xEnsFull = np.zeros((ensLen, xSize[0], xSize[1], xSize[2]))
    for iens in range(ensLen):
        xEnsFull[iens] = np.array(ncList[iens][varName])
    return xEnsFull

def collectEnsemble_zInterp(ncList, varName, zMode, zVal):
    ensLen = len(ncList)
    domainSize = np.array(ncList[0]['P']).shape[1:]
    xEnsFullInMass = np.zeros((ensLen, domainSize[0], domainSize[1], domainSize[2]))
    xEnsAtZVal = np.zeros((ensLen, domainSize[1], domainSize[2]))

    for iens in range(ensLen):
        xEnsFullInMass[iens] = interpToMassGrid3D(np.array(ncList[iens][varName])[0, :, :, :],
                                                  ncList[0][varName].stagger)

    if zMode == 'mlev':
        xEnsAtZVal = xEnsFullInMass[:, zVal, :, :]
    elif zMode == 'plev':
        for iens in range(ensLen):
            pressure = ((np.array(ncList[iens]['P'])+np.array(ncList[iens]['PB']))[0, :, :, :])/100
            xEnsAtZVal[iens] = interplevel(xEnsFullInMass[iens], pressure, zVal, meta=False, missing=0)
    elif zMode == 'hlev':
        for iens in range(ensLen):
            height = ((np.array(ncList[iens]['PH'])+np.array(ncList[iens]['PHB']))[0, :, :, :])/9.8
            heightInMassGrid = interpToMassGrid3D(height, 'Z')
            xEnsAtZVal[iens] = interplevel(xEnsFullInMass[iens], heightInMassGrid, zVal, meta=False, missing=0)
    else:
        raise ValueError('zMode should be mlev, plev, or hlev')
    del (xEnsFullInMass)
    return xEnsAtZVal

def interpToMassGrid2D(data, staggerType):
    if staggerType == 'X':
        dataInMassGrid = (data[:, 0:-1]+data[:, 1:])/2.
    elif staggerType == 'Y':
        dataInMassGrid = (data[0:-1, :]+data[1:, :])/2.
    elif staggerType == '':
        dataInMassGrid = data
    else:
        raise ValueError('staggerType should be X or Y')
    return dataInMassGrid

def interpToMassGrid3D(data, staggerType):
    if staggerType == 'X':
        dataInMassGrid = (data[:, :, 0:-1]+data[:, :, 1:])/2.
    elif staggerType == 'Y':
        dataInMassGrid = (data[:, 0:-1, :]+data[:, 1:, :])/2.
    elif staggerType == 'Z':
        dataInMassGrid = (data[0:-1, :, :]+data[1:, :, :])/2.
    elif staggerType == '':
        dataInMassGrid = data
    else:
        raise ValueError('staggerType should be X, Y, or Z')
    return dataInMassGrid
