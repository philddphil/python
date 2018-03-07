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
f0 = (r"C:\Users\Philip\Documents\Technical Stuff\Hologram optimisation"
      r"\Algorithmic implementation\180227\Post realignment\Port 2\fibre2.csv")
f1 = p1 + r'\Phase Ps.csv'
f2 = p1 + r'\Phase greys.csv'
fibre = 2
fibre_c = 'fibre9d_' + str(fibre)
holo_data = np.genfromtxt(f0, delimiter=',')
print(holo_data)

Λ = holo_data[0]
# Λ = 10
φ = (np.pi / 180) * holo_data[1]
H_δx = int(holo_data[4])
H_δy = int(holo_data[5])
ϕ_lw = π * holo_data[10]
# ϕ_lw = π * 0.5
ϕ_up = π * holo_data[11]
# ϕ_up = π * 2.5

os_lw = π * holo_data[12]
# os_lw = π * 0.3
os_up = π * holo_data[13]
# os_up = π * 0.3
osw_lw = holo_data[14]
# osw_lw = 1
osw_up = holo_data[15]
# osw_up = 10

off = holo_data[16]

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
ϕ_g_lu = prd.ϕ_g_fun(x3, popt[0], popt[1])
ϕ_g = interp1d(np.linspace(0, 255, 256), ϕ_g_lu)

ϕ_max = ϕ_g_lu[-1]
ϕ1 = np.linspace(0, ϕ_max, 256)


X = range(H_δx)
Y = range(H_δy)
Z1 = prd.phase_tilt(Λ, φ, H_δx, H_δy, ϕ_lw, ϕ_up, off)
Z1a = prd.phase_tilt(Λ, π / 2, H_δx, H_δy, ϕ_lw, ϕ_up, off)
Z2 = prd.phase_sin(Λ, φ, H_δx, H_δy, ϕ_lw, ϕ_up, off, 0.5, 0)
Z1_mod = prd.phase_mod(Z1, ϕ_lw, ϕ_up)
Z1a_mod = prd.phase_mod(Z1a, ϕ_lw, ϕ_up)


g_ϕ0 = interp1d(ϕ_g_lu, np.linspace(0, 255, 256))

gs0 = g_ϕ0(ϕ1)

g_ind1 = gs0 < g_ϕ0(ϕ_lw + os_lw)
g_ind2 = gs0 > g_ϕ0(ϕ_up - os_up)

gs1 = copy.copy(gs0)
gs2 = copy.copy(gs0)
gs1[g_ind1] = 0
gs2[g_ind2] = 255

gs1b = prd.n_G_blurs(gs1, osw_lw)
gs2b = prd.n_G_blurs(gs2, osw_up)
g_mid = int(g_ϕ0((ϕ_up - ϕ_lw) / 2 + ϕ_lw))

gs3 = np.concatenate((gs1b[0:g_mid], gs2b[g_mid:]))
gs4 = np.concatenate((gs1[0:g_mid], gs2[g_mid:]))

g_ϕ1 = interp1d(ϕ1, gs3)

Z12_mod = prd.phase_mod(Z1 + Z2, ϕ_lw, ϕ_up)
H1 = prd.remap_phase(Z1_mod, g_ϕ0)
H1a = prd.remap_phase(Z1a_mod, g_ϕ0)
H2 = prd.remap_phase(Z1_mod, g_ϕ1)
H2a = prd.remap_phase(Z1a_mod, g_ϕ1)

Profile1 = ϕ_g(H2a[Λ + 1:2 * Λ + 1, 0]) - min(ϕ_g(H2a[Λ + 1:2 * Λ + 1, 0]))
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


fig1 = plt.figure('fig1', figsize=(4, 4))
ax1 = fig1.add_subplot(1, 1, 1)
fig1.patch.set_facecolor(cs['mdk_dgrey'])
ax1.set_ylabel('y axis - phase')
ax1.set_xlabel('x axis - pixel')
plt.plot(H2a[:, 0], '.:', c=cs[fibre_c])
plt.tight_layout()

# l3 = plt.plot(Z1a_mod[0, :], '.:')

fig2 = plt.figure('fig2', figsize=(4, 4))
ax2 = fig2.add_subplot(1, 1, 1)
fig2.patch.set_facecolor(cs['mdk_dgrey'])
plt.plot(ϕ1 / π, gs0)
plt.plot(ϕ1 / π, gs3)
# plt.plot(ϕ1/π, gs4)
ax2.set_ylabel('y axis - greylevel')
ax2.set_xlabel('x axis - phase')
plt.tight_layout()


fig3 = plt.figure('fig3', figsize=(4, 4))
ax3 = fig3.add_subplot(1, 1, 1)
fig3.patch.set_facecolor(cs['mdk_dgrey'])
plt.plot(Profile1/π, '.:', c=cs[fibre_c])
ax3.set_ylabel('phase - / π')
ax3.set_xlabel('x axis - px')

plt.tight_layout()

fig4 = plt.figure('fig4', figsize=(4, 4))
ax4 = fig4.add_subplot(1, 1, 1)
fig4.patch.set_facecolor(cs['mdk_dgrey'])
ax4.set_ylabel('y axis - px')
ax4.set_xlabel('x axis - px')
l5 = plt.imshow(H2[:, :], cmap='binary')

# l6 = plt.plot(ϕ1 / π, g_ϕ1(ϕ1), '.')


# l6 = plt.plot(ϕ1 / π, g_ϕ3(ϕ1))

# datacursor(l1, bbox=dict(fc=cs['mdk_yellow'], alpha=1))
# datacursor(l4, bbox=dict(fc=cs['mdk_yellow'], alpha=1))
plt.tight_layout()
plt.show()
os.chdir(p1)
prd.PPT_save_2d(fig1, ax1, 'plot1.png')
prd.PPT_save_2d(fig2, ax2, 'plot2.png')
prd.PPT_save_2d(fig3, ax3, 'plot3.png')
prd.PPT_save_2d(fig4, ax4, 'plot4.png')
