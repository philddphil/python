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
π = np.pi
p1 = (r"C:\Users\Philip\Documents\Technical Stuff\Hologram optimisation"
      r"\Single param sweep\180208\2nd sweep (fixed comma prob)")
f0 = r"\Sweep H.txt"
f1 = r"\Swept i0.txt"
f2 = r"\Swept IL.txt"
f3 = r"\Swept MF.txt"
f4 = r"\Swept XT.txt"
f5 = r"\Swept param.txt"

i0s = np.genfromtxt(p1 + f1, delimiter=',')
IL = np.genfromtxt(p1 + f2, delimiter=',')
MF = np.genfromtxt(p1 + f3, delimiter=',')
XT = np.genfromtxt(p1 + f4, delimiter=',')
v = np.genfromtxt(p1 + f5, delimiter=',')
print(np.mean(IL,0))
print(XT)
print(MF)
print(IL)

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
#                     '.', cmap='gray', s=6, c=Z1_mod)
# ggred = scat0.get_facecolor()

# cm = plt.get_cmap('binary')
# surf0 = ax0.plot_surface(X[:, 0:a], Y[:, 0:a], Z1[
#                          :, 0:a], cmap='gray', alpha=0.6)
# wire0 = ax0.plot_wireframe(X[:, 0:a], Y[:, 0:a], Z1[
#     :, 0:a], color=cs['mdk_dgrey'], lw=0.5, alpha=1)

fig1 = plt.figure('fig1', figsize=(3, 3))
ax1 = fig1.add_subplot(1, 1, 1)
fig1.patch.set_facecolor(cs['mdk_dgrey'])
ax1.set_xlabel('x axis - min phase / π')
# ax1.set_ylabel('y axis - phase/π')
ax1.set_ylabel('y axis - a.u')

# ax1.set_xlabel('x axis - g')
# ax1.set_ylabel('y axis - P')

plt.plot(v, MF-np.mean(MF), label = 'MF')
plt.plot(v, IL-np.mean(IL), label = 'IL')
plt.plot(v, XT-np.mean(XT), label = 'XT')
plt.legend()


# plt.plot(H1[0, :], 'o:')
# plt.plot(H2[0, :], 'o:')
# plt.ylim(0, 255)
# plt.plot(Z2[0, :] / π, 'o:')
# plt.plot(ϕ1, 'o:')
# plt.plot(Z12_mod[0, :] / π, 'o:')
# plt.ylim(-1, 2)

# plt.imshow(Z12_mod, extent=prd.extents(X) + prd.extents(Y))
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
os.chdir(f2)
prd.PPT_save_2d(fig1, ax1, 'plot1.png')
