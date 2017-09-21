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

from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import interp1d
from scipy.signal import find_peaks_cwt
from scipy.ndimage.filters import gaussian_filter
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import savgol_filter


##############################################################################
# Import some extra special libraries from my own repo
##############################################################################
sys.path.insert(0, r'C:\Users\Philip\Documents\Python\Local Repo\library')
import useful_defs_prd as prd
from peakdetect import peakdetect
cs = prd.palette()

##############################################################################
# Do some stuff
##############################################################################



##############################################################################
# Plot some figures
##############################################################################

fig0 = plt.figure('fig0')
ax0 = Axes3D(fig0)
fig0.patch.set_facecolor(cs['mdk_dgrey'])
ax0.w_xaxis.set_pane_color(cs['mdk_dgrey'])
ax0.w_yaxis.set_pane_color(cs['mdk_dgrey'])
ax0.w_zaxis.set_pane_color(cs['mdk_dgrey'])
ax0.set_xlabel('x axis')
ax0.set_ylabel('y axis')
ax0.set_zlabel('z axis')

fig1 = plt.figure('fig1')
ax1 = fig1.add_subplot(1, 1, 1)
fig1.patch.set_facecolor(cs['mdk_dgrey'])
ax1.set_xlabel('x axis')
ax1.set_ylabel('y axis')

im2 = plt.figure('im2')
ax2 = im2.add_subplot(1, 1, 1)
im2.patch.set_facecolor(cs['mdk_dgrey'])
ax2.set_xlabel('x axis')
ax2.set_ylabel('y axis')
cb2 = plt.colorbar()


plt.show()