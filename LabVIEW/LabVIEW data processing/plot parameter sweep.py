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
f1 = r"C:\Users\User\Documents\Phils Data\Data files\171004\Phi_lwlim_1\Ps.csv"
p1 = r"C:\Users\User\Documents\Phils Data\Data files\171004\Phi_lwlim_1"
os.chdir(p1)
d1 = np.genfromtxt(f1, delimiter=',')
print(np.shape(d1))


##############################################################################
# Plot some figures
##############################################################################


fig1 = plt.figure('fig1')
ax1 = fig1.add_subplot(1, 1, 1)
fig1.patch.set_facecolor(cs['mdk_dgrey'])
ax1.set_xlabel('x axis')
ax1.set_ylabel('y axis')
plt.plot(d1[:, 1], '.--', lw=0.5, label= 'Ins. Loss (dB)')
plt.plot(np.abs(d1[:, 2] - d1[:, 1]), '.--', lw=0.5, label = 'X-talk (dB)')
leg1 = plt.legend()
leg1.get_frame().set_alpha(0.1)

plt.show()

prd.PPT_save_2d(fig1, ax1, 'figure1')
