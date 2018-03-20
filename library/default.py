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

from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import interp1d
from scipy.signal import find_peaks_cwt
from scipy.ndimage.filters import gaussian_filter
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import savgol_filter
from matplotlib import cm

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
p0 = (r"C:\Users\Philip\Documents\Technical Stuff\Hologram optimisation"
      r"\Replay field calculation\180320\Camera replay field study")


f0 = p0 + r"\10 min blazed.csv"
x_off = 525
y_off = 619

im, coords = prd.img_csv(f0)
im1 = im - np.min(np.min(im))
X = coords[0]
Y = coords[1]
x = X[0, :] - x_off
y = Y[:, 0] - y_off
max_im = prd.max_i_2d(im)

Profile = im[max_im[0], :]
Smoothed_prof = prd.n_G_blurs(Profile, 3)

max_im = prd.max_i_2d(im)
print(max_im)
##############################################################################
# Plot some figures
##############################################################################

fig1 = plt.figure('fig1', figsize=(5, 5))
ax1 = fig1.add_subplot(1, 1, 1)
fig1.patch.set_facecolor(cs['mdk_dgrey'])
ax1.set_xlabel('x axis')
ax1.set_ylabel('y axis')
plt.imshow(im, extent=prd.extents(x) + prd.extents(y))

fig2 = plt.figure('fig2', figsize=(5, 5))
ax2 = fig2.add_subplot(1, 1, 1)
fig2.patch.set_facecolor(cs['mdk_dgrey'])
ax2.set_xlabel('x axis')
ax2.set_ylabel('y axis')
plt.plot(Profile, '.--', lw=0.5)
plt.plot(Smoothed_prof)

os.chdir(p0)
plt.tight_layout()
plt.show()
prd.PPT_save_2d(fig1, ax1, 'plot1.png')
