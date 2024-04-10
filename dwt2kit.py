import pywt
import numpy as np

def signal_decomp(data, waveName, times, mode=pywt.Modes.constant):
    w = pywt.Wavelet(waveName)
    cA = np.zeros(data.shape)
    cA[:] = data
    cACollc = []  # approximation
    cDCollc = []  # horizontal detail [horizontal, vertical, diagonal]
    for i in range(times):
        cA, (cH, cV, cD) = pywt.dwt2(cA, w, mode)
        cACollc.append(cA)
        cDCollc.append((cH, cV, cD))
    return cACollc, cDCollc

def signal_recomp(cACollc, cDCollc, w, mode=pywt.Modes.constant):
    rec_a = []
    rec_d = []

    for i, coeff in enumerate(cACollc):
        coeff_list = [coeff, (None, None, None)]+[(None, None, None)]*i
        rec_a.append(pywt.waverec2(coeff_list, w, mode))

    for i, coeff in enumerate(cDCollc):
        coeff_list = [None, coeff]+[(None, None, None)]*i
        rec_d.append(pywt.waverec2(coeff_list, w, mode))

    return rec_a, rec_d

def qk_wlt_decomposition(data, wave, alllevs):
    cACollc, cDCollc = signal_decomp(data, wave, alllevs)
    data_a, data_d = signal_recomp(cACollc, cDCollc, wave)
    return data_a, data_d

def calc_residual(data_raw, data_a, data_d):
    yLen, xLen = data_raw.shape
    residual = []
    residual.append(data_raw[0:yLen, 0:xLen]-data_a[0][0:yLen, 0:xLen]-data_d[0][0:yLen, 0:xLen])

    for ilev in range(1, len(data_a)):
        full = data_a[ilev-1][0:yLen, 0:xLen]
        estimate = data_a[ilev][0:yLen, 0:xLen]+data_d[ilev][0:yLen, 0:xLen]
        residual.append(full-estimate)
    return residual

def combine_into_scales(coef_scaleFromLev, data_raw, data_a, data_d, aToScale=0, replaceResidual=True):
    data_in_each_scale = []
    scaLen = len(coef_scaleFromLev)
    levLen = len(coef_scaleFromLev[0])
    yLen, xLen = data_raw.shape

    if replaceResidual:
        residual = calc_residual(data_raw, data_a, data_d)
        for ilev in range(len(data_a)):
            data_d[ilev][0:yLen, 0:xLen] = data_d[ilev][0:yLen, 0:xLen]+residual[ilev][0:yLen, 0:xLen]
    for iSca in range(scaLen):
        if aToScale == iSca:
            dataTemp = data_a[levLen-1][0:yLen, 0:xLen]
        else:
            dataTemp = np.zeros(data_raw.shape)

        for ilev in range(levLen):
            dataTemp = dataTemp+data_d[ilev][0:yLen, 0:xLen]*coef_scaleFromLev[iSca][ilev]

        data_in_each_scale.append(dataTemp)

    data_residual = data_raw-np.sum(data_in_each_scale, axis=0)

    return data_in_each_scale, data_residual

def sdi_compose(coef, data_raw, data_a, data_d):
    levLen = len(coef)
    yLen, xLen = data_raw.shape
    residual = calc_residual(data_raw, data_a, data_d)

    for ilev in range(len(data_a)):
        data_d[ilev][0:yLen, 0:xLen] = data_d[ilev][0:yLen, 0:xLen]+residual[ilev][0:yLen, 0:xLen]
    dataInflated = data_a[levLen-1][0:yLen, 0:xLen]

    for ilev in range(levLen):
        dataInflated = dataInflated+data_d[ilev][0:yLen, 0:xLen]*coef[ilev]

    return dataInflated
