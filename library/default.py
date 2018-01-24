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
<<<<<<< HEAD
H = prd.holo_tilt()[1]
R = prd.holo_replay()
print(np.shape(H), np.shape(R))
fig1 = plt.figure()
plt.imshow(H)
fig2 = plt.figure()
plt.imshow(R)
=======
# read in image files from path p1
<<<<<<< HEAD
x = np.linspace(-1, 4, 500)
t = np.linspace(0, 499, 500)
=======
res = 1000
R = 10
F = 12
x = np.linspace(-F, F, res)
y = np.linspace(-F, F, res)
coords = np.meshgrid(x, y)

g0 = prd.Gaussian_2D(coords, 1, 0, 5, 1, 1)
g1 = g0.reshape(res, res)

g0 = prd.Gaussian_2D(coords, 1, 0, -5, 1, 1)
g2 = g0.reshape(res, res)

G = g1 + g2

η1 = sp.trapz(sp.trapz((g1 * g2), y), x)**2
η2 = sp.trapz(sp.trapz(g1**2, y), x) * sp.trapz(sp.trapz(g2**2, y), x)
print(η1 / η2)
>>>>>>> 364446309de5edc8cc868af64eff55c37b380735

y = (1 - sp.special.erf(x)) / 2

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
<<<<<<< HEAD
plt.plot(t, y)
=======
ax1.set_aspect(1)

# plt.imshow(G, extent=(x[0], x[-1], y[0], y[-1]), origin='lower')
surffit = ax1.contour(*coords, G, 5	, cmap=cm.jet)

plt.plot(xc, yc)
>>>>>>> 364446309de5edc8cc868af64eff55c37b380735

# im3 = plt.figure('im3')
# ax3 = im3.add_subplot(1, 1, 1)
# im3.patch.set_facecolor(cs['mdk_dgrey'])
# ax3.set_xlabel('x axis')
# ax3.set_ylabel('y axis')
# plt.imshow(im)
# cb2 = plt.colorbar()
# plt.legend()

>>>>>>> f6a0ca6db27842bf3e78028267523e68f8f1f741
plt.show()
