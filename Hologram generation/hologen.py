##############################################################################
# Import some libraries
##############################################################################

import random
import os
import glob
import copy
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as io
import pylab as pl
import scipy.optimize as opt
import scipy.misc

from scipy.interpolate import interp1d

###############################################################################
# Define some functions
###############################################################################


# Generate holograms with first two parameters to optimise - Λ and φ
def Holo_tilt(Λ, φ, Hol_δy, Hol_δx, ϕ_min, ϕ_max):
    x = np.arange(Hol_δx)
    y = np.arange(Hol_δy)
    [X, Y] = np.meshgrid(x, y)

    θ = np.arctan((ϕ_max - ϕ_min) / Λ)

    Z = np.tan(θ) * (X * np.cos(φ) + Y * np.sin(φ))

    Z_mod = Z % (ϕ_max - ϕ_min - 0.00000001)
    Z_mod = Z_mod * (ϕ_max - ϕ_min) / (np.max(Z_mod)) + ϕ_min
    Holo_s = (Z, Z_mod)
    return Holo_s


# Add sub hologram Z_mod to larger hologram (initially set to 0s)
def Add_Holo(Hol_cy, Hol_cx, Z_mod, LCOSy, LCOSx):
    Holo_f = np.zeros((LCOSy, LCOSx))
    (Hol_δy, Hol_δx) = np.shape(Z_mod)
    y1 = np.int(Hol_cy - np.floor(Hol_δy / 2))
    y2 = np.int(Hol_cy + np.ceil(Hol_δy / 2))
    x1 = np.int(Hol_cx - np.floor(Hol_δx / 2))
    x2 = np.int(Hol_cx + np.ceil(Hol_δx / 2))
    Holo_f[y1:y2, x1:x2] = Z_mod
    return Holo_f


# Defining the functional form of grayscale to phase (g(ϕ))
def Phase(x, A, B):
    ϕ = np.square(np.sin(A * (1 - np.exp(-B * x))))
    return ϕ


# Use g(ϕ) defined in 'Phase' to fit experimentally obtained phaseramps
def Fit_phase():
    p1 = r"C:\Users\Philip\Documents\Python\Local Repo\Curve fitting"
    os.chdir(p1)
    files = glob.glob('*Phaseramp.mat')
    phaseramp = io.loadmat(files[0])
    y_dB = phaseramp['P4'].ravel()
    y_lin = np.power(10, y_dB / 10) / np.max(np.power(10, y_dB / 10))
    x0 = np.linspace(0, 255, len(y_dB))
    x1 = np.linspace(0, 255, 25)
    x3 = range(255)
    f1 = interp1d(x0, y_lin)
    initial_guess = (15, 1 / 800)

    try:
        popt, pcov = opt.curve_fit(Phase, x1, f1(
            x1), p0=initial_guess, bounds=([0, -np.inf], [np.inf, np.inf]))

    except RuntimeError:
        print("Error - curve_fit failed")

    ϕ_A = popt[0]
    ϕ_B = popt[1]
    ϕ_g = (2 / np.pi) * np.abs(ϕ_A) * (1 - np.exp(-ϕ_B * x3))

    return (ϕ_A, ϕ_B, ϕ_g)


# Use the fitting results from 'Fit_phase' & linear mapping
# to remap hologram Z_mod
def Remap_phase(Z_mod, g_ϕ):
    Z_mod1 = copy.copy(Z_mod)
    for i1 in range(np.shape(Z_mod)[0]):
        Z_mod1[i1, :] = g_ϕ(Z_mod[i1, :])
    return (Z_mod1)


# Save bmp file
def Save_holo(Hologram, Path):
    scipy.misc.imsave(Path, Hologram)


# Overshoot mapping
def Overshoot_phase(Z_mod1, g_OSlw, g_OSup, g_min, g_max):

    Z_mod2 = copy.copy(Z_mod1)
    Super_thres_indices = Z_mod1 > g_OSup
    Sub_thres_indices = Z_mod1 <= g_OSlw
    Z_mod2[Super_thres_indices] = g_max
    Z_mod2[Sub_thres_indices] = g_min

    return Z_mod2


###############################################################################
# Calculate phase map
###############################################################################

# Phase mapping details (ϕ)
(ϕ_A, ϕ_B, ϕ_g) = Fit_phase()
ϕ_min = 0
ϕ_max = max(ϕ_g)
print(ϕ_max)
ϕ_rng = (ϕ_min, ϕ_max)
g_ϕ = interp1d(ϕ_g, range(255))

###############################################################################
# Specify parameters
###############################################################################

# LCOS size (# pixels in x & y)
LCOS_δy = 100
LCOS_δx = 100

# Subhologram size and location
Hol_δx = 50
Hol_δy = 50
Hol_cx = int(LCOS_δx / 2)
Hol_cy = int(LCOS_δy / 2)

# Phase (ϕ) upper and lower limits
ϕ_uplim = ϕ_max
ϕ_lwlim = ϕ_min

# Overshooting thresholds
g_OSup = g_ϕ(2)
g_OSlw = g_ϕ(0.2)
# g_OSup = g_ϕ(ϕ_max/2)
# g_OSlw = g_ϕ(ϕ_max/2)
g_min = 0
g_max = 255

# Grating metrics (specify Λ (period) and φ (rotation angle))
Λ = 7.5
φ = 0.25 * np.pi

###############################################################################
# Construct some stuff for the code
###############################################################################
LCOS_δyx = (LCOS_δy, LCOS_δx)
Hol_δyx = (Hol_δy, Hol_δx)
Hol_cyx = (Hol_cy, Hol_cx)
ϕ_lims = (ϕ_lwlim, ϕ_uplim)

# Define holo params
Holo_params = (Λ, φ, *Hol_δyx, *φ_lims)

###############################################################################
# Run some of the functions defined above
###############################################################################

# Calculate sub hologram (Holo_s)
Holo_s = Holo_tilt(*Holo_params)
Z = Holo_s[0]
Zϕ_mod = Holo_s[1]

# Remap phase with non linear ϕ map
Zg_mod1 = Remap_phase(Zϕ_mod, g_ϕ)

# Use overshooting
Zg_mod2 = Overshoot_phase(Zg_mod1, g_OSlw, g_OSup, g_min, g_max)

# Calculate full holograms (Holo_fN)
Holo_f1 = Add_Holo(*Hol_cyx, Zg_mod1, *LCOS_δyx)
Holo_f2 = Add_Holo(*Hol_cyx, Zg_mod2, *LCOS_δyx)

# Set output holograms (Z_out, Holo_out)
Z_out = Zg_mod2
Holo_out = Holo_f2

###############################################################################
# Save output
###############################################################################
Save_holo(Holo_out, 'Name.bmp')

###############################################################################
# Plotting
###############################################################################
cmap = plt.get_cmap('gray')

###############################################################################

# # Plot Z_mod
# pl.figure('Z_mod')
# im0 = plt.imshow(Z_mod, cmap)

###############################################################################

# pl.figure('g vs x')
# pl.plot(range(Hol_δy), g2_ϕ(Z_mod[:, 0]), ':b.')
# pl.plot(range(Hol_δy), g1_ϕ(Z_mod[:, 0]), ':r.')
# pl.title('g vs x position')

###############################################################################
# pl.figure('g vs ϕ')
# pl.plot(g2_ϕ(Z_mod[:, 0]), ':b.')
# pl.plot(g1_ϕ(Z_mod[:, 0]), ':r.')
# pl.title('g vs ϕ position')

###############################################################################
# pl.figure('Full Phase')
# im0 = plt.imshow(Holo_f0, cmap)
# pl.title('Λ = %s, ϕ = %sπ' % (Λ, φ / np.pi))
# plt.colorbar()
# plt.clim(ϕ_min, ϕ_max)
# print(np.max(Holo_f0))

###############################################################################
# Full Holograms
pl.figure('Full Hologram1')
im0 = plt.imshow(Holo_f2, cmap)
pl.title('Λ = %s, ϕ = %sπ' % (Λ, φ / np.pi))
plt.colorbar()
plt.clim(0, 255)
plt.ylabel('LCOS y axis')
plt.xlabel('LCOS x axis')

###############################################################################
# Phase Profile
# pl.figure('Phase Profile')
# plt.plot(range(LCOS_δy), ϕ_lwlim *
#          np.ones(np.shape(Holo_f0[:, Hol_cx])), 'xkcd:green')
# plt.plot(range(LCOS_δy), ϕ_uplim *
#          np.ones(np.shape(Holo_f0[:, Hol_cx])), 'xkcd:green')
# plt.plot(range(LCOS_δy), Holo_f0[:, Hol_cx], '.:',
#          color='xkcd:light green', mfc='xkcd:dark green')

# pl.title('Λ = %s, ϕ = %sπ' % (Λ, φ / np.pi))
# plt.ylim([ϕ_min, ϕ_max])
# plt.grid()

###############################################################################
# Hologram profiles (x/y)
pl.figure('Hologram Profiles')
plt.plot(range(LCOS_δx), g_ϕ(ϕ_lwlim) *
         np.ones(np.shape(Holo_out[:, Hol_cx])), '--', color='xkcd:light blue')
plt.plot(range(LCOS_δx), g_ϕ(ϕ_uplim) *
         np.ones(np.shape(Holo_out[:, Hol_cx])), '--', color='xkcd:light blue')

plt.plot(range(LCOS_δx), Holo_out[:, Hol_cx], '.-',
         color='xkcd:blue', mfc='xkcd:dark blue', label='Horizontal profile')

plt.plot(range(LCOS_δy), g_ϕ(ϕ_lwlim) *
         np.ones(np.shape(Holo_out[Hol_cy, :])), ':', color='xkcd:light red')
plt.plot(range(LCOS_δy), g_ϕ(ϕ_uplim) *
         np.ones(np.shape(Holo_out[Hol_cy, :])), ':', color='xkcd:light red')

plt.plot(range(LCOS_δy), Holo_out[Hol_cy, :], '.-',
         color='xkcd:red', mfc='xkcd:dark red', label='Vertical profile')

pl.title('Λ = %s, ϕ = %sπ' % (Λ, φ / np.pi))
plt.ylim([0, 255])
plt.ylabel('Greyscale value [0:255]')
plt.xlabel('LCOS y axis')
plt.legend()
plt.grid()

###############################################################################
# pl.figure('ϕ mapping')
# pl.plot(range(255), ϕ_g, color='xkcd:red', label='From exp. φ mapping')
# pl.plot(g2_ϕ(ϕ_lwlim) * np.ones(100),
#         np.linspace(ϕ_min, ϕ_lwlim, 100), ':', color='xkcd:light red')
# pl.plot(g2_ϕ(ϕ_uplim) * np.ones(100),
#         np.linspace(ϕ_min, ϕ_uplim, 100), ':', color='xkcd:light red')

# pl.plot(range(255), np.linspace(ϕ_min, ϕ_max, 255),
#         color='xkcd:blue', label='Linear mapping')
# pl.plot(g1_ϕ(ϕ_lwlim) * np.ones(100),
#         np.linspace(ϕ_min, ϕ_lwlim, 100), ':', color='xkcd:light blue')
# pl.plot(g1_ϕ(ϕ_uplim) * np.ones(100),
#         np.linspace(ϕ_min, ϕ_uplim, 100), ':', color='xkcd:light blue')
# pl.plot(range(255), ϕ_uplim * np.ones(255), color='xkcd:black')
# pl.plot(range(255), ϕ_lwlim * np.ones(255), color='xkcd:black')

# plt.legend()
# plt.ylabel('Phase (φ)')
# plt.xlabel('Greyscale value [0:255]')

plt.show()
