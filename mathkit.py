import numpy as np
from math import log10


def ceil_nthDigit(num, n):
    return np.ceil(num/(10**np.floor(log10(num)-(n-1))))*(10**np.floor(log10(num)-(n-1)))


def round_nthDigit(num, n):
    return np.round(num/(10**np.floor(log10(num)-(n-1))))*(10**np.floor(log10(num)-(n-1)))


def floor_nthDigit(num, n):
    return np.floor(num/(10**np.floor(log10(num)-(n-1))))*(10**np.floor(log10(num)-(n-1)))


def vec2diagMat(vec):
    diagMat = np.zeros([vec.size, vec.size])
    for ix in range(vec.size):
        diagMat[ix, ix] = vec[ix]
    return diagMat

def vec2posDiagMat(vec):
    vec[vec<0] = 0
    diagMat = np.zeros([vec.size, vec.size])
    for ix in range(vec.size):
        diagMat[ix, ix] = vec[ix]
    return diagMat

def calc_sqrtMat_by_eigh(mat):
    # cholesky is about 10-times faster than eigh, but the result is unusable
    [w, v] = np.linalg.eigh(mat)
    w[w<0] = 0
    wSqrtMat = np.zeros(mat.shape)
    for ix in range(mat.shape[0]):
        wSqrtMat[ix, ix] = np.sqrt(w[ix])
    sqrtMat = v @ wSqrtMat @ v.T
    return sqrtMat

def calc_invMat_by_eigh(mat):
    [w, v] = np.linalg.eigh(mat)
    wInvMat = vec2diagMat(w**-1)
    invMat = v @ wInvMat @ v.T
    return invMat

def calc_invSqrtMat_by_eigh(mat):
    [w, v] = np.linalg.eigh(mat)
    w[w<0] = 0
    wInvSqrtMat = np.zeros(mat.shape)
    for ix in range(mat.shape[0]):
        wInvSqrtMat[ix, ix] = (np.sqrt(w[ix]))**(-1)
    invSqrtMat = v @ wInvSqrtMat @ v.T
    return invSqrtMat

def rms(val): return np.sqrt(np.mean(val**2))
def ms(val): return np.mean(val**2)
def sumAbs(val): return np.sum(np.abs(val))

def findPolyRoot_NewtonsMethod(func, dfunc, xInit, maxIter=100, threshold=10**-4):
    for i in range(maxIter):
        changement = func(xInit)/dfunc(xInit)
        xInit = xInit-changement
        if np.abs(func(xInit)) <= threshold:
            break
        if np.abs(changement) <= threshold:
            print('findPolyRoot_NewtonsMethod: error stop decaying, exit loop, error might be still large')
            break
    if i == maxIter-1:
        print('findPolyRoot_NewtonsMethod: reach maxIter, error might be still large')
    return xInit
