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
f1 = r"C:\Users\Philip\Documents\LabVIEW\Data\Calibration files\Phaseramp.mat"
f2 = r"C:\Users\Philip\Desktop"
Λ = 20
φ = 0 * π / 4
H_δx = 2 * Λ
H_δy = 2 * Λ
ϕ_lw = 0.5 * π
ϕ_up = 1.5 * π
off = Λ


files = glob.glob(f1)
phaseramp = io.loadmat(files[0])

y_dB = phaseramp['P4'].ravel()
y_lin = np.power(10, y_dB / 10) / np.max(np.power(10, y_dB / 10))

<<<<<<< HEAD

x0 = np.linspace(0, 255, len(y_dB))
x1 = np.linspace(0, 255, 25)
x3 = np.linspace(0, 255, 255)
f1 = interp1d(x0, y_lin)
initial_guess = (15, 1 / 800)

popt, _ = opt.curve_fit(prd.P_phase, x1, f1(x1), p0=initial_guess,
                        bounds=([0, -np.inf], [np.inf, np.inf]))
print(popt)
Ps = prd.P_phase(x3, popt[0], popt[1])
ϕ_A = popt[0]
ϕ_B = popt[1]

X = range(H_δx)
Y = range(H_δy)
Z1 = prd.phase_tilt(Λ, φ, H_δx, H_δy, ϕ_lw, ϕ_up, off)
Z2 = prd.phase_sin(Λ, φ, H_δx, H_δy, ϕ_lw, ϕ_up, off, 1, 0)
Z1_mod = prd.phase_mod(Z1, ϕ_lw, ϕ_up)
Z12_mod = prd.phase_mod(Z1 + Z2, ϕ_lw, ϕ_up)
# ϕ_g = (2 / np.pi) * np.abs(ϕ_A) * (1 - np.exp(-ϕ_B * x))

=======
x = np.linspace(0,255,255)
P = prd.P_phase(x, 15, 1/800)
>>>>>>> 716447576f12cc3460335ac650dfc73d29afdbfc
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

fig1 = plt.figure('fig1', figsize=(4, 4))
ax1 = fig1.add_subplot(1, 1, 1)
fig1.patch.set_facecolor(cs['mdk_dgrey'])
<<<<<<< HEAD
ax1.set_xlabel('x axis - px')
ax1.set_ylabel('y axis - phase/π')
=======
ax1.set_xlabel('x axis - g')
ax1.set_ylabel('y axis - P')

plt.plot(x, P)

>>>>>>> 716447576f12cc3460335ac650dfc73d29afdbfc

plt.plot(Z1_mod[0, :] / π, 'o:')
plt.plot(Z2[0, :] / π, 'o:')
plt.plot(Z12_mod[0, :] / π, 'o:')
plt.ylim(-1, 2)
# plt.imshow(Z1_mod, extent=prd.extents(X) + prd.extents(Y))
# plt.colorbar()
# im3 = plt.figure('im3')
# ax3 = im3.add_subplot(1, 1, 1)
# im3.patch.set_facecolor(cs['mdk_dgrey'])
# ax3.set_xlabel('x axis')
# ax3.set_ylabel('y axis')
# plt.imshow(im)
# cb2 = plt.colorbar()
# plt.legend()

plt.show()
os.chdir(f2)
prd.PPT_save_2d(fig1, ax1, 'plot1.png')
