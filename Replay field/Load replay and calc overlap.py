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
      r"\Replay field calculation\180313\From post re-alignment 180227\Port 2")
f0 = p0 + r'\*.csv'
files = glob.glob(f0)

I = np.genfromtxt(p0 + r'\I.csv', delimiter=',')
x = np.genfromtxt(p0 + r'\x.csv', delimiter=',')
y = np.genfromtxt(p0 + r'\y.csv', delimiter=',')
coords = np.meshgrid(x, y)


lwy = 30
lwx = 30

port1 = prd.Gaussian_2D(coords, 1, -220, 220, lwx, lwy)
port1 = port1.reshape(len(x), len(y))

port2 = prd.Gaussian_2D(coords, 1, 0, 220, lwx, lwy)
port2 = port2.reshape(len(x), len(y))

port3 = prd.Gaussian_2D(coords, 1, 220, 220, lwx, lwy)
port3 = port3.reshape(len(x), len(y))

port4 = prd.Gaussian_2D(coords, 1, -220, 0, lwx, lwy)
port4 = port4.reshape(len(x), len(y))

port5 = prd.Gaussian_2D(coords, 1, 0, 0, lwx, lwy)
port5 = port5.reshape(len(x), len(y))

port6 = prd.Gaussian_2D(coords, 1, 220, 0, lwx, lwy)
port6 = port6.reshape(len(x), len(y))

port7 = prd.Gaussian_2D(coords, 1, -220, -220, lwx, lwy)
port7 = port7.reshape(len(x), len(y))

port8 = prd.Gaussian_2D(coords, 1, 0, -220, lwx, lwy)
port8 = port8.reshape(len(x), len(y))

port9 = prd.Gaussian_2D(coords, 1, 220, -220, lwx, lwy)
port9 = port9.reshape(len(x), len(y))

ports = (port1 + port2 + port3
         + port4 + port5 + port6
         + port7 + port8 + port9)

η1 = 10 * np.log10(prd.overlap(x, y, I, port1))
η2 = 10 * np.log10(prd.overlap(x, y, I, port2))
η3 = 10 * np.log10(prd.overlap(x, y, I, port3))
η4 = 10 * np.log10(prd.overlap(x, y, I, port4))
η6 = 10 * np.log10(prd.overlap(x, y, I, port6))
η5 = 10 * np.log10(prd.overlap(x, y, I, port5))
η7 = 10 * np.log10(prd.overlap(x, y, I, port7))
η8 = 10 * np.log10(prd.overlap(x, y, I, port8))
η9 = 10 * np.log10(prd.overlap(x, y, I, port9))

ηs = [η1, η2, η3, η4, η6, η7, η8, η9]
ηs = ηs - max(ηs)
ηs_sorted = sorted(ηs)
port_numbers = [1, 2, 3, 4, 6, 7, 8, 9]
print(ηs_sorted[-2])

fig1 = plt.figure('fig1', figsize=(4, 4))
ax1 = fig1.add_subplot(1, 1, 1)
fig1.patch.set_facecolor(cs['mdk_dgrey'])
ax1.set_ylabel('y axis - μm')
ax1.set_xlabel('x axis - μm')
# plt.imshow(ports, extent=prd.extents(x) +
#            prd.extents(y), origin='bottom', cmap='viridis')
plt.imshow(10 * np.log10(I), extent=prd.extents(x) +
           prd.extents(y), origin='bottom', vmin=-80, vmax=0)
ax1.contour(x, y, ports, 8,
            colors='w', alpha=0.7, linewidths=0.5)
fig1.tight_layout()
# plt.axis('off')

fig2 = plt.figure('fig2', figsize=(4, 4))
ax2 = fig2.add_subplot(1, 1, 1)
fig2.patch.set_facecolor(cs['mdk_dgrey'])
ax2.set_ylabel('power - dB')
ax2.set_xlabel('x axis - μm')
plt.plot(x, 10 * np.log10(I[:, 100]))
fig2.tight_layout()

fig3 = plt.figure('fig3', figsize=(4, 4))
ax3 = fig3.add_subplot(1, 1, 1)
fig3.patch.set_facecolor(cs['mdk_dgrey'])
ax3.set_ylabel('power - dB')
ax3.set_xlabel('port number')
plt.plot(port_numbers, ηs, 'o')
fig3.tight_layout()

plt.show()
os.chdir(p0)
prd.PPT_save_2d(fig3, ax3, 'XTs.png')
prd.PPT_save_2d(fig2, ax2, 'Profile.png')
prd.PPT_save_2d(fig1, ax1, 'ports.png')
