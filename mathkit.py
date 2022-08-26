import numpy as np
from math import log10


def ceil_nthDigit(num, n):
    return np.ceil(num/(10**np.floor(log10(num)-(n-1))))*(10**np.floor(log10(num)-(n-1)))
