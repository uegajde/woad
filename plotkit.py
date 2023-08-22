import matplotlib.colors as colors
from matplotlib.cm import get_cmap
import numpy as np


class MidpointNormalize(colors.Normalize):
    # ref: https://matplotlib.org/stable/gallery/userdemo/colormap_normalizations.html
    def __init__(self, vmin=None, vmax=None, vcenter=None, clip=False):
        self.vcenter = vcenter
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def unfoldLon(lon, center=180):
    # shift the longitude discontinuity to center+180°
    # e.g. when center=180 (default),
    #      the discontinuity is at 0°E and it is better for figure foucing on 180°

    newlon = np.zeros(lon.shape)
    newlon[:] = lon[:]
    newlon[lon < center-180] = lon[lon < center-180]+360
    newlon[lon > center+180] = lon[lon > center+180]-360

    return newlon


def cal_subplot_positions(rows, cols, marginTop=0.08, marginButton=0.05, marginLeft=0.08, marginRight=0.05, verticalGap=0.05, horizontalGap=0.05):
    axHeight = (1-marginTop-marginButton-(rows-1)*verticalGap)/rows
    axWidth = (1-marginLeft-marginRight-(cols-1)*horizontalGap)/cols

    positions = []
    for irow in range(0, rows):
        for icol in range(0, cols):
            xStart = marginLeft+icol*(horizontalGap+axWidth)
            yStart = marginButton+(cols-irow-1)*(verticalGap+axHeight)
            positions.append([xStart, yStart, axWidth, axHeight])

    return positions


def cmap_cwb_radar_reflec():
    cmap_name = 'cwb_radar_reflec'
    cmap_colors = [(0, 255/255, 255/255), (0, 236/255, 255/255), (0, 218/255, 255/255), (0, 200/255, 255/255), (0, 182/255, 255/255),  # 0-5
                   (0, 163/255, 255/255), (0, 145/255, 255/255), (0, 127/255, 255/255), (0, 109/255, 255/255), (0, 91/255, 255/255),  # 5-10
                   (0, 72/255, 255/255), (0, 54/255, 255/255), (0, 36/255, 255/255), (0, 18/255, 255/255), (0, 0, 255/255),  # 10-15
                   (0, 255/255, 0), (0, 244/255, 0), (0, 233/255, 0), (0, 222/255, 0), (0, 211/255, 0),  # 15-20
                   (0, 200/255, 0), (0, 190/255, 0), (0, 180/255, 0), (0, 170/255, 0), (0, 160/255, 0),  # 20-25
                   (0, 150/255, 0), (0, 171/255, 0), (102/255, 192/255, 0), (153/255, 213/255, 0), (204/255, 234/255, 0),  # 25-30
                   (255/255, 255/255, 0), (255/255, 244/255, 0), (255/255, 233/255, 0), (255/255, 222/255, 0), (255/255, 211/255, 0),  # 30-35
                   (255/255, 200/255, 0), (255/255, 184/255, 0), (255/255, 168/255, 0), (255/255, 152/255, 0),  (255/255, 136/255, 0),  # 35-40
                   (255/255, 120/255, 0), (255/255, 96/255, 0), (255/255, 72/255, 0), (255/255, 48/255, 0), (255/255, 24/255, 0),  # 40-45
                   (255/255, 0, 0), (244/255, 0, 0), (233/255, 0, 0), (222/255, 0, 0), (211/255, 0, 0),  # 45-50
                   (200/255, 0, 0), (190/255, 0, 0), (180/255, 0, 0), (170/255, 0, 0), (160/255, 0, 0),  # 50-55
                   (150/255, 0, 0), (171/255, 0, 51/255), (192/255, 0, 102/255), (213/255, 0, 153/255),  (234/255, 0, 204/255),  # 55-60
                   (255/255, 0, 255/255), (234/255, 0, 255/255), (213/255, 0, 255/255), (192/255, 0, 255/255), (171/255, 0, 255/255)]  # 60-65
    cmap = colors.LinearSegmentedColormap.from_list(cmap_name, cmap_colors, N=65)
    cmap.set_over((150/255, 0, 255/255))
    cmap.set_under((1, 1, 1))
    norm = colors.Normalize(vmin=0, vmax=65)
    return cmap, norm


def cmap_cwb_satellite_ctt_enhanced():
    cmap_name = 'cwb_satellite_ctt_enhanced'
    temp_nodes = [28, 23, 0, -20, -25, -31, -42, -53, -59, -63, -80, -110, -136]

    rStart = np.array([0, 0, 0, 200, 255, 255, 255, 0, 0, 255, 255, 255])/255
    rEnd = np.array([0, 0, 0, 200, 255, 255, 255, 0, 0, 255, 255, 255])/255
    gStart = np.array([0, 45, 145, 150, 255, 0, 255, 0, 200, 0, 255, 57])/255
    gEnd = np.array([0, 180, 255, 150, 255, 0, 255, 0, 200, 255, 255, 58])/255
    bStart = np.array([200, 255, 0, 0, 0, 0, 255, 0, 200, 255, 255, 0])/255
    bEnd = np.array([255, 255, 0, 0, 0, 0, 255, 0, 200, 255, 255, 252])/255

    cmap_colors = []
    for icc in range(0, len(temp_nodes)-1):
        length = temp_nodes[icc]-temp_nodes[icc+1]
        icc_temp = np.ones((length, 4))
        icc_temp[:, 0] = np.linspace(rStart[icc], rEnd[icc], length)
        icc_temp[:, 1] = np.linspace(gStart[icc], gEnd[icc], length)
        icc_temp[:, 2] = np.linspace(bStart[icc], bEnd[icc], length)
        cmap_colors.append(icc_temp)

    cmap_colors = np.concatenate(cmap_colors)
    cmap_colors = cmap_colors[::-1]
    cmap = colors.LinearSegmentedColormap.from_list(cmap_name, cmap_colors, N=164)
    norm = colors.Normalize(vmin=-136, vmax=28)
    return cmap, norm


def cmap_jaxa_rainfall_rate():
    cmap_name = 'jaxa_rainfall_rate'
    cmap_colors = [(0, 0, 150/255),  # 0.0-0.5
                   (0, 100/255, 1),  # 0.5-1.0
                   (0, 180/255, 1),  # 1.0-2.0
                   (51/255, 219/255, 128/255),  # 2.0-3.0
                   (155/255, 235/255, 74/255),  # 3.0-5.0
                   (1, 235/255, 0),  # 5.0-10.
                   (1, 179/255, 0),  # 10.-15.
                   (1, 100/255, 0),  # 15.-20.
                   (235/255, 30/255, 0),  # 20.-25.
                   (175/255, 0, 0)]  # 25.-30.]
    cmap = colors.LinearSegmentedColormap.from_list(cmap_name, cmap_colors, N=10)
    clevels = np.array([0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
    norm = colors.BoundaryNorm(clevels, len(clevels)-1)
    return cmap, norm


def cmap_cwb_rainfall():
    cmap_name = 'cwb_rainfall'
    cmap_colors = [(202/255, 202/255, 202/255),  # 0-1
                   (158/255, 253/255, 255/255),  # 1-2
                   (1/255, 210/255, 253/255),  # 2-6
                   (0/255, 165/255, 254/255),  # 6-10
                   (1/255, 119/255, 253/255),  # 10-15
                   (38/255, 163/255, 27/255),  # 15-20
                   (1/255, 249/255, 47/255),  # 20-30
                   (255/255, 254/255, 50/255),  # 30-40
                   (255/255, 211/255, 40/255),  # 40-50
                   (255/255, 167/255, 31/255),  # 50-70
                   (255/255, 43/255, 6/255),  # 70-90
                   (218/255, 35/255, 4/255),  # 90-110
                   (170/255, 24/255, 1/255),  # 110-130
                   (169/255, 34/255, 163/255),  # 130-150
                   (220/255, 45/255, 210/255),  # 150-200
                   (255/255, 56/255, 251/255)]  # 200-300
    cmap = colors.LinearSegmentedColormap.from_list(cmap_name, cmap_colors, N=16)
    cmap.set_over((255/255, 213/255, 253/255))
    cmap.set_under((1, 1, 1))
    clevels = np.array([0, 1, 2, 6, 10, 15, 20, 30, 40, 50, 70, 90, 110, 130, 150, 200, 300])
    norm = colors.BoundaryNorm(clevels, len(clevels)-1)
    return cmap, norm


def cmap_jaxa_water_vapor():
    cmap = get_cmap('jet')
    norm = colors.Normalize(vmin=0, vmax=70)

    return cmap, norm
