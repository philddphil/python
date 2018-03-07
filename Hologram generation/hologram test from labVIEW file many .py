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

fig1 = plt.figure('fig3', figsize=(4, 4))
ax1 = fig1.add_subplot(1, 1, 1)
fig1.patch.set_facecolor(cs['mdk_dgrey'])
ax1.set_ylabel('phase - / π')
ax1.set_xlabel('x axis - px')
# prd.holo_load(files[1], p1)

for i1, val1 in enumerate(files):
    print(val1)
    prd.holo_load(val1, p1)

plt.show()
os.chdir(p0)
prd.PPT_save_2d(fig1, ax1, 'plot1.png')
