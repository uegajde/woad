import numpy as np

nauticalmile2km = 1.852
knot2ms = 0.514444444
earth_radius = 6378137.0  # unit: m
denyStr = ['no', 'n', 'nope', 'f', 'false', '0']


def gravitational_acceleration(lat):
    # formula ref: https://en.wikipedia.org/wiki/Gravitational_acceleration
    g_polos = 9.832
    g_45 = 9.806
    g_equator = 9.780

    g = g_45-(g_polos-g_equator)*np.cos(2*np.deg2rad(lat))/2
    return g


def coriolis_coefficient(lat):
    # formula ref: https://en.wikipedia.org/wiki/Coriolis_frequency
    rotation_rate = 7.2921 * 10**-5
    f = 2*rotation_rate*np.sin(np.deg2rad(lat))
    return f
