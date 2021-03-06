##############################################################################
# Import some libraries
##############################################################################
import sys
import os
import glob
import time
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import csv
import scipy.optimize as opt
import socket
import scipy.io as io
import importlib.util
import ntpath
import random

from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import interp1d
from scipy.signal import find_peaks_cwt
from scipy.ndimage.filters import gaussian_filter
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import savgol_filter


##############################################################################
# Import some extra special libraries from my own repo
##############################################################################
# sys.path.insert(0, r'C:\Users\Philip\Documents\Python\Local Repo\library')
import useful_defs_prd as prd
from peakdetect import peakdetect
cs = prd.palette()

##############################################################################
# Do some stuff
##############################################################################

Map_p = 'Map.txt'
Hc1_p = 'Holocs1.txt'
Hc2_p = 'Holocs2.txt'
Hd_p = 'Holod'

# f0 = open('i0.txt', 'w')
# f0.write(str(0))
# f0.close()

# f1 = open('i1.txt', 'w')
# f1.write(str(1))
# f1.close()

# Map = []
# Hc1 = []
# Hc2 = []
# Hd = []
# np.savetxt(Hc2_p, Hc2, fmt='%f', delimiter=',')

Map = np.genfromtxt(Map_p, dtype='int', delimiter=',')
Hc1 = np.genfromtxt(Hc1_p, delimiter=',')
Hc2 = np.genfromtxt(Hc2_p, delimiter=',')
Hd = np.genfromtxt(Hd_p, delimiter=',')

Map = np.atleast_1d(Map)
Hc1 = np.atleast_1d(Hc1)
Hc2 = np.atleast_1d(Hc2)
Hd = np.atleast_1d(Hd)

i0 = np.genfromtxt('i0.txt', dtype='int')
i1 = np.genfromtxt('i1.txt', dtype='int')

dc = (1 / (2 ** (i1 + 1)))
hd = 1 / 2**(i1)

print('i1 = ', i1)
print('i0 = ', i0)
start = 0.5
shift = 0


if i1 == 1:
    start = 0.5
    print('Start shift = ', 0)
else:
    for j1 in range(i1 - 1):
        print(Map[j1], ' ', 1 / (2**(j1 + 2)), -
              (1 / (2**(j1 + 2))) * (-1)**(Map[j1]))
        shift = -(1 / (2**(j1 + 2))) * (-1)**(Map[j1]) + shift
    start = 0.5 + shift
    print('Start shift = ', shift)

print('Start = ', start)
if i0 == 0:
    Hc1 = np.append(Hc1, start - dc)
    mp = random.randint(0, 1)
    Map = np.append(Map, mp)

    Hd = np.append(Hd, hd)
    np.savetxt(Map_p, Map, fmt='%d', delimiter=',')
    np.savetxt(Hc1_p, Hc1, fmt='%.4f', delimiter=',')
    np.savetxt(Hd_p, Hd, fmt='%.4f', delimiter=',')

    f0 = open('i0.txt', 'w')
    f0.write(str(1))
    f0.close()

elif i0 == 1:
    Hc2 = np.append(Hc2, start + dc)
    np.savetxt(Hc2_p, Hc2, fmt='%.4f', delimiter=',')
    f0 = open('i0.txt', 'w')
    f0.write(str(0))
    f0.close()
    f1 = open('i1.txt', 'w')
    f1.write(str(i1 + 1))
    f1.close()
    print('Map = ', Map, type(Map))

    fig1 = plt.figure('fig1')
    ax1 = fig1.add_subplot(1, 1, 1)
    fig1.patch.set_facecolor(cs['mdk_dgrey'])
    ax1.set_xlabel('x axis')
    ax1.set_ylabel('y axis')

    plt.plot(Hc1 - Hd / 2, '-', c='xkcd:pale red')
    plt.plot(Hc1 + Hd / 2, '-', c='xkcd:pale red')
    plt.plot(Hc2 - Hd / 2, ':', c='xkcd:light blue')
    plt.plot(Hc2 + Hd / 2, '-', c='xkcd:light blue')

    for j2, val2 in enumerate(Map):
        print(j2, val2)
        if val2 == 1:
            c1 = 'red'
            c2 = 'green'
        else:
            c2 = 'red'
            c1 = 'green'

        plt.plot(j2, Hc2[j2], 'o', c=c2)
        plt.plot(j2, Hc1[j2], 'o', c=c1)

    plt.show()


print('Hc1:', Hc1)
print('Hc2:', Hc2)
print('Hd:', Hd)


# elif i0 == 1:
# 	hc = 1/2 + 1 / (2 ** (i0 + 1))

##############################################################################
# Plot some figures
##############################################################################
