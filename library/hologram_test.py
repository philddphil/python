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
f1 = r"C:\Users\Philip\Desktop\fibre1.csv"
data = np.genfromtxt(f1, delimiter=',')
os.chdir(r"C:\Users\Philip\Documents\LabVIEW\labview-python\python-code")
params = prd.variable_unpack(data)
params[2] = 100
params[3] = 100
params[8] = 0.2
params[9] = 2.0
params[10] = 15
params[11] = 180
params[12] = 0
params[13] = 255
params[14] = 25
params[15] = 0.6 * np.pi / 2
params[16] = 10
params[17] = 0.1
params[18] = 0
[H1, H2, H3, H4, H5, H6] = prd.holo_gen(*params)


##############################################################################
# Plot some figures
##############################################################################
a = 5

fig1 = plt.figure('fig1')
ax1 = fig1.add_subplot(1, 1, 1)
fig1.patch.set_facecolor(cs['mdk_dgrey'])
ax1.set_xlabel('x axis')
ax1.set_ylabel('y axis')

p1 = plt.plot(H1[:, 0], '.--', lw=0.5)
p2 = plt.plot(H2[:, 0], '.--', lw=0.5)

ggred = p1[0].get_color()
ggblue = p2[0].get_color()

fig2 = plt.figure('fig2')
ax2 = fig2.add_subplot(1, 1, 1)
fig2.patch.set_facecolor(cs['mdk_dgrey'])
ax2.set_xlabel('x axis')
ax2.set_ylabel('y axis')

p1 = plt.plot(H3[:, 0], '.--', lw=0.5)
p2 = plt.plot(H4[:, 0], '.--', lw=0.5)

# ggred = p1[0].get_color()
# ggblue = p2[0].get_color()

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
#                     '.', cmap='gray', s=16, c=Z1[:, 0:a])

# surf0 = ax0.plot_surface(X[:, 0:a], Y[:, 0:a], Z1[
#                          :, 0:a], cmap='gray', alpha=0.1, edgecolor=ggred)


im2 = plt.figure('im2')
ax2 = im2.add_subplot(1, 1, 1)
im2.patch.set_facecolor(cs['mdk_dgrey'])
ax2.set_xlabel('x axis')
ax2.set_ylabel('y axis')
plt.imshow(H5, cmap='gray')
cb2 = plt.colorbar()

im3 = plt.figure('im3')
ax3 = im3.add_subplot(1, 1, 1)
im3.patch.set_facecolor(cs['mdk_dgrey'])
ax3.set_xlabel('x axis')
ax3.set_ylabel('y axis')
plt.imshow(H6, cmap='gray')
cb3 = plt.colorbar()


# fig2 = plt.figure('fig2')
# ax2 = Axes3D(fig2)
# fig2.patch.set_facecolor(cs['mdk_dgrey'])
# ax2.w_xaxis.set_pane_color(cs['mdk_dgrey'])
# ax2.w_yaxis.set_pane_color(cs['mdk_dgrey'])
# ax2.w_zaxis.set_pane_color(cs['mdk_dgrey'])
# ax2.set_xlabel('x axis')
# ax2.set_ylabel('y axis')
# ax2.set_zlabel('z axis')
# scat2 = ax2.scatter(X, Y, Z0,
#                     '.', cmap='gray', s=16, c=Z0)
# surf2 = ax2.plot_surface(X, Y, Z0, cmap='gray', alpha=0.1, edgecolor=ggred)

plt.show()
