##############################################################################
# Import some libraries
##############################################################################
import sys
import os
import glob
import time
import re
import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
import csv
import scipy.optimize as opt
import socket
import scipy as sp
import scipy.io as io
import importlib.util
import ntpath
import copy

from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import interp1d
from scipy.signal import find_peaks_cwt
from scipy.ndimage.filters import gaussian_filter
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import savgol_filter
from matplotlib import cm
from scipy.special import erf
from mpldatacursor import datacursor


##############################################################################
# Import some extra special libraries from my own repo and do some other stuff
##############################################################################
sys.path.insert(0, r'C:\Users\Philip\Documents\Python\Local Repo\library')
np.set_printoptions(suppress=True)
import useful_defs_prd as prd
from peakdetect import peakdetect
cs = prd.palette()


##############################################################################
# Do some stuff
##############################################################################
π = np.pi

x = np.linspace(-100, 100, 500)
y = x
coords = np.meshgrid(x, y)

port1 = prd.Gaussian_2D(coords, 1, 50, 0, 5, 5)
port1 = np.reshape(port1, np.shape(coords[0]))

port2 = prd.Gaussian_2D(coords, 1, -50, 0, 5, 5)
port2 = np.reshape(port2, np.shape(coords[0]))

diff1 = prd.Gaussian_2D(coords, 1, -50, 0, 5, 5)
diff1 = np.reshape(diff1, np.shape(coords[0]))

diff2 = prd.Gaussian_2D(coords, 0.1, 50, 0, 5, 5)
diff2 = np.reshape(diff2, np.shape(coords[0]))

ports = port1 + port2
diff_all = diff1 + diff2
diff_dB = 10 * np.log10(diff_all**2)
diff_dB[diff_dB < -60] = -60

η1 = prd.Overlap(x, y, port1, diff_all)
η2 = prd.Overlap(x, y, port2, diff_all)

print('η1 = ', η1)
print('η2 = ', η2)
print(10 * np.log10(η1 / η2))


fig1 = plt.figure('fig1', figsize=(4, 4))
ax1 = fig1.add_subplot(1, 1, 1)
fig1.patch.set_facecolor(cs['mdk_dgrey'])
ax1.set_ylabel('y axis')
ax1.set_xlabel('x axis')
plt.imshow(diff1 + diff2, extent=prd.extents(x) + prd.extents(y))
ax1.contour(x, y, port2 + port1, 5, colors='w', alpha=0.3, linewidths=0.5)


fig2 = plt.figure('fig2', figsize=(4, 4))
ax2 = fig2.add_subplot(1, 1, 1)
fig2.patch.set_facecolor(cs['mdk_dgrey'])
ax2.set_ylabel('y axis')
ax2.set_xlabel('x axis')
plt.plot(diff_dB[250, :])


plt.show()
