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
p1 = (r"C:\Users\Philip\Documents\Technical Stuff\Hologram optimisation"
      r"\Algorithmic implementation\180226 By port")

p0 = (r"C:\Users\Philip\Documents\Technical Stuff\Hologram optimisation"
      r"\Algorithmic implementation\180227\Post realignment\All holos")
f0 = p0 + r'\*.csv'
files = glob.glob(f0)

fig1 = plt.figure('fig1', figsize=(4, 4))
ax1 = fig1.add_subplot(1, 1, 1)
fig1.patch.set_facecolor(cs['mdk_dgrey'])
ax1.set_ylabel('phase - / π')
ax1.set_xlabel('x axis - px')

fig2 = plt.figure('fig2', figsize=(4, 4))
ax2 = fig2.add_subplot(1, 1, 1)
fig2.patch.set_facecolor(cs['mdk_dgrey'])
ax2.set_ylabel('y axis - μm')
ax2.set_xlabel('x axis - μm')

fig3 = plt.figure('fig3', figsize=(4, 4))
ax3 = fig3.add_subplot(1, 1, 1)
fig3.patch.set_facecolor(cs['mdk_dgrey'])
ax3.set_ylabel('power - dB')
ax3.set_xlabel('x axis - μm')

fig4 = plt.figure('fig4', figsize=(4, 4))
fig4.patch.set_facecolor(cs['mdk_dgrey'])
ax4 = fig4.add_subplot(1, 1, 1)
ax4.set_ylabel('y axis - μm')
ax4.set_xlabel('x axis - μm')

fig5 = plt.figure('fig5', figsize=(4, 4))
fig5.patch.set_facecolor(cs['mdk_dgrey'])
ax5 = fig5.add_subplot(1, 1, 1)
ax5.set_ylabel('y axis - μm')
ax5.set_xlabel('x axis - μm')

I, x, y = prd.holo_replay(files[1], p1)
np.savetxt(p0 + r'\I.csv', I, delimiter=',')
np.savetxt(p0 + r'\x.csv', x, delimiter=',')
np.savetxt(p0 + r'\y.csv', y, delimiter=',')
plt.show()
os.chdir(p0)
prd.PPT_save_2d(fig1, ax1, 'plot1.png')
