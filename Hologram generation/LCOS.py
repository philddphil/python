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
# Define pixel array
os.chdir(r"C:\Users\Philip\Documents\LabVIEW\labview-python\python-code")
LCOS_δx = 1920
LCOS_δy = 1080

Hol_δx = 50
Hol_δy = 50
Hol_cx = 960
Hol_cy = 540

ϕ_min = 0
ϕ_max = 2.5
ϕ_lwlim = 0
ϕ_uplim = 2

g_OSlw = 20
g_OSup = 205
g_min = 0
g_max = 255

Λ = 10
φ = 90
offset = 0

sin_amp = 0
sin_off = 0

params = [LCOS_δx, LCOS_δy,
          Hol_δx, Hol_δy, Hol_cx, Hol_cy,
          ϕ_min, ϕ_max, ϕ_lwlim, ϕ_uplim,
          g_OSlw, g_OSup, g_min, g_max,
          Λ, φ, offset, sin_amp, sin_off]

[Z1_p, Z2_p, H1_1_p, H3_1_p, H1_1, H3_1] = prd.holo_gen(*params)
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
ax1.set_xlabel('x axis (px)')
ax1.set_ylabel('Phase (ϕ) axis')
plt.plot(Z1_p[:,0], '.--')
plt.plot(Z2_p[:,0], '.--')

fig2 = plt.figure('fig2')
ax2 = fig2.add_subplot(1, 1, 1)
fig2.patch.set_facecolor(cs['mdk_dgrey'])
ax2.set_xlabel('x axis (px)')
ax2.set_ylabel('Hologram (g) axis')
plt.plot(H3_1_p[:,0], '.--')
plt.plot(H1_1_p[:,0], '.--')

im3 = plt.figure('im3')
ax3 = im3.add_subplot(1, 1, 1)
im3.patch.set_facecolor(cs['mdk_dgrey'])
ax3.set_xlabel('x axis')
ax3.set_ylabel('y axis')
plt.imshow(H1_1)
cb2 = plt.colorbar()
plt.legend()

plt.show()
