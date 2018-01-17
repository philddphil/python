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
δ = 2
pad = 3

D0 = np.ones((δ, δ))

D1 = np.zeros(((2 * pad + 1) * (δ), (2 * pad + 1) * (δ)))

D2 = D1
x = np.shape(D2)[0]
y = np.shape(D2)[1]
fig1 = plt.figure('fig1')
fig1.patch.set_facecolor(cs['mdk_dgrey'])
ax1 = fig1.add_subplot(1, 1, 1)
ax1.set_xlabel('x axis')
ax1.set_ylabel('y axis')
for i1 in range(x):
    for i2 in range(y):
        if (i1 - pad) % (2 * pad + 1) == 0 and (i2 - pad) % (2 * pad + 1) == 0:
            print((i1 - pad) // (2 * pad + 1), (i2 - pad) // (2 * pad + 1))
            plt.plot(i1, i2, 'o', c=cs['ggred'])
        else:
            plt.plot(i1, i2, 's', c=cs['ggblue'])

D3 = prd.Pad_A_elements(D0, pad)


fig2 = plt.figure('fig2')
fig2.patch.set_facecolor(cs['mdk_dgrey'])
ax2 = fig2.add_subplot(1, 1, 1)
plt.imshow(D3)
ax2.set_xlabel('x axis')
ax2.set_ylabel('y axis')

plt.show()

prd.PPT_save_2d(fig1, ax1, 'plot.png')
prd.PPT_save_2d(fig2, ax2, 'img.png')

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

# fig1 = plt.figure('fig1')
# ax1 = fig1.add_subplot(1, 1, 1)
# fig1.patch.set_facecolor(cs['mdk_dgrey'])
# ax1.set_xlabel('x axis')
# ax1.set_ylabel('y axis')
# ax1.set_aspect(1)

# # plt.imshow(G, extent=(x[0], x[-1], y[0], y[-1]), origin='lower')
# surffit = ax1.contour(*coords, G, 5	, cmap=cm.jet)

# plt.plot(xc, yc)

# im3 = plt.figure('im3')
# ax3 = im3.add_subplot(1, 1, 1)
# im3.patch.set_facecolor(cs['mdk_dgrey'])
# ax3.set_xlabel('x axis')
# ax3.set_ylabel('y axis')
# plt.imshow(im)
# cb2 = plt.colorbar()
# plt.legend()

plt.show()
