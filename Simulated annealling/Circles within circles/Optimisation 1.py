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
# read in image files from path p1
res = 1000
R = 3
F = 8
r1 = 2
r2 = 2 * np.sqrt(3)
r3 = 4
r4 = 3 * np.sqrt(3)
π = np.pi
n = 2
# first 13 Hex coordinates ###################################################
Hex_coords = [(0, 0)]
for i1 in range(6 * n):
    a0 = i1 + 1
    a1 = int(np.ceil((i1 + 1) / 6))
    a2 = i1 % 6

    if a1 % 2 == 1:
        ϕ = a2 * π / 3
        r = a1 + 1
    else:
        ϕ = a2 * π / 3 + π / 6
        r = (1 + a1 / 2) * np.sqrt(3)
    
    Hex_coords.append((r, ϕ))


x = np.linspace(-F, F, res)
y = np.linspace(-F, F, res)
coords = np.meshgrid(x, y)
Gtot = np.zeros((res, res))

for i1, val in enumerate(Hex_coords):
    x_c = val[0] * np.cos(val[1])
    y_c = val[0] * np.sin(val[1])
    g = prd.Gaussian_2D(coords, 1, x_c, y_c, 0.3, 0.3)
    G = g.reshape(res, res)

    Gtot = Gtot + G


(xc, yc) = prd.circle(R, 0, 0)


##############################################################################
# Plot some figures
##############################################################################

# a = 50

# fig0 = plt.figure('fig0')
# ax0 = Axes3D(fig0)
# fig0.patch.set_facecolor(cs['mdk_dgrey'])
# ax0.w_xaxis.set_pane_color(cs['mdk_dgrey'])
# ax0.w_yaxis.set_pane_color(cs['mdk_dgrey'])
# ax0.w_zaxis.set_pane_color(cs['mdk_dgrey'])
# ax0.set_xlabel('x axis')
# ax0.set_ylabel('y axis')
# ax0.set_zlabel('z axis')
# scat0 = ax0.scatter(X[:, 0:a], Y[:, 0:a], Z1[:, 0:a],
#                     '.', cmap='gray', s=6, c=Z1)
# ggred = scat0.get_facecolor()

# cm = plt.get_cmap('binary')
# surf0 = ax0.plot_surface(X[:, 0:a], Y[:, 0:a], Z1[
#                          :, 0:a], cmap='gray', alpha=0.6)
# wire0 = ax0.plot_wireframe(X[:, 0:a], Y[:, 0:a], Z1[
#     :, 0:a], color=cs['mdk_dgrey'], lw=0.5, alpha=1)

fig1 = plt.figure('fig1')
ax1 = fig1.add_subplot(1, 1, 1)
fig1.patch.set_facecolor(cs['mdk_dgrey'])
ax1.set_xlabel('x axis')
ax1.set_ylabel('y axis')
ax1.set_aspect(1)

plt.imshow(Gtot, extent=(x[0], x[-1], y[0], y[-1]), origin='lower')
# surffit = ax1.contour(*coords, G, 5   , cmap=cm.jet)

plt.plot(xc, yc)

# im3 = plt.figure('im3')
# ax3 = im3.add_subplot(1, 1, 1)
# im3.patch.set_facecolor(cs['mdk_dgrey'])
# ax3.set_xlabel('x axis')
# ax3.set_ylabel('y axis')
# plt.imshow(im)
# cb2 = plt.colorbar()
# plt.legend()
plt.tight_layout()
plt.show()
prd.PPT_save_2d(fig1,ax1,'Hex pack.png')
