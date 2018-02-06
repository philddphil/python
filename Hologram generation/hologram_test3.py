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
p1 = (r"C:\Users\Philip\Documents\Technical Stuff"
      r"\Hologram optimisation\Polynomial distortion")
f1 = p1 + r'\Phase Ps.csv'
f2 = p1 + r'\Phase greys.csv'
Λ = 10
φ = 0 * π / 4
H_δx = 40
H_δy = 40
ϕ_lw = 0.5 * π
ϕ_up = 2.5 * π
off = 0
g_OSlw = 0
g_OSup = 255
g_min = 0
g_max = 255


y_dB = np.genfromtxt(f1, delimiter=',')
y_lin = np.power(10, y_dB / 10) / np.max(np.power(10, y_dB / 10))

x0 = np.genfromtxt(f2, delimiter=',')
x1 = np.linspace(0, 255, 25)
x3 = np.linspace(0, 255, 256)
f1 = interp1d(x0, y_lin)
initial_guess = (15, 1 / 800)

try:
    popt, _ = opt.curve_fit(prd.P_g_fun, x1, f1(
        x1), p0=initial_guess, bounds=([0, -np.inf], [np.inf, np.inf]))

except RuntimeError:
    print("Error - curve_fit failed")
ϕ_A = popt[0]
ϕ_B = popt[1]
ϕ_g = prd.ϕ_g_fun(x3, popt[0], popt[1])
g_ϕ = interp1d(ϕ_g, np.linspace(0, 255, 256))

ϕ_max = ϕ_g[-1]
ϕ1 = np.linspace(0, ϕ_max, 256)


X = range(H_δx)
Y = range(H_δy)
Z1 = prd.phase_tilt(Λ, φ, H_δx, H_δy, ϕ_lw, ϕ_up, off)
Z2 = prd.phase_sin(Λ, φ, H_δx, H_δy, ϕ_lw, ϕ_up, off, 0.5, 0)
Z1_mod = prd.phase_mod(Z1, ϕ_lw, ϕ_up)


g_ϕ0 = interp1d(ϕ_g, np.linspace(0, 255, 256))

gs0 = g_ϕ0(ϕ1)

g_ind1 = gs0 < g_ϕ(ϕ_lw + 0.1)
g_ind2 = gs0 > g_ϕ(ϕ_up - 0.5)

gs0[g_ind1] = 0
gs0[g_ind2] = g_max

gs0 = prd.n_G_blurs(gs0, 0.5)

g_ϕ1 = interp1d(ϕ1, gs0)

Z12_mod = prd.phase_mod(Z1 + Z2, ϕ_lw, ϕ_up)
H1 = prd.remap_phase(Z1_mod, g_ϕ)
H2 = prd.remap_phase(Z1_mod, g_ϕ1)

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

# fig1 = plt.figure('fig1', figsize=(4, 4))
# ax1 = fig1.add_subplot(1, 1, 1)
# fig1.patch.set_facecolor(cs['mdk_dgrey'])
# ax1.set_ylabel('y axis - phase ϕ')
# l0 = plt.plot(ϕ_g / π)
# l1 = plt.plot(ϕ_g1 / π)
# l2 = plt.plot(ϕ_g2 / π)


# ax1.set_xlabel('x axis - greylevel')


fig2 = plt.figure('fig2', figsize=(4, 4))
ax2 = fig2.add_subplot(1, 1, 1)
fig2.patch.set_facecolor(cs['mdk_dgrey'])
ax2.set_ylabel('y axis - greylevel')
ax2.set_xlabel('x axis - pixel')
l3 = plt.plot(H1[0, :], '.:')
l4 = plt.plot(H2[0, :], '.:')

fig3 = plt.figure('fig3', figsize=(4, 4))
ax3 = fig3.add_subplot(1, 1, 1)
fig3.patch.set_facecolor(cs['mdk_dgrey'])
ax3.set_ylabel('y axis - greylevel')
ax3.set_xlabel('x axis - phase ϕ')

l5 = plt.plot(ϕ1 / π, g_ϕ(ϕ1), '.')
l6 = plt.plot(ϕ1 / π, g_ϕ1(ϕ1), '.')


# l6 = plt.plot(ϕ1 / π, g_ϕ3(ϕ1))

# datacursor(l1, bbox=dict(fc=cs['mdk_yellow'], alpha=1))
# datacursor(l4, bbox=dict(fc=cs['mdk_yellow'], alpha=1))
plt.tight_layout()
plt.show()
os.chdir(p1)
# prd.PPT_save_2d(fig1, ax1, 'plot1.png')
