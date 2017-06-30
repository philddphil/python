
import random
import os
import glob
import copy
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as io
import pylab as pl
import scipy.optimize as opt

from scipy.interpolate import interp1d

###############################################################################
# Define some functions
###############################################################################


def Holo_tilt(Λ, φ, Hol_δy, Hol_δx, ϕ_min, ϕ_max):
    x = np.arange(Hol_δx)
    y = np.arange(Hol_δy)
    θ = np.arctan((ϕ_max - ϕ_min) / Λ)
    [X, Y] = np.meshgrid(x, y)
    Z = np.tan(θ) * (X * np.cos(φ) + Y * np.sin(φ))
    Z_mod = Z % (ϕ_max - ϕ_min)
    Z_mod = Z_mod * (ϕ_max - ϕ_min) / (np.max(Z_mod)) + ϕ_min
    Holo_s = (Z, Z_mod)
    return Holo_s


def Add_Holo(Hol_cy, Hol_cx, Z_mod, LCOSy, LCOSx):
    Holo_f = np.zeros((LCOSy, LCOSx))
    (Hol_δy, Hol_δx) = np.shape(Z_mod)
    y1 = np.int(Hol_cy - np.floor(Hol_δy / 2))
    y2 = np.int(Hol_cy + np.ceil(Hol_δy / 2))
    x1 = np.int(Hol_cx - np.floor(Hol_δx / 2))
    x2 = np.int(Hol_cx + np.ceil(Hol_δx / 2))
    Holo_f[y1:y2, x1:x2] = Z_mod
    return Holo_f


def Phase(x, A, B):
    P = np.square(np.sin(A * (1 - np.exp(-B * x))))
    return P


def Fit_phase():
    p1 = r'C:\Users\Philip\Documents\Python files\curve fitting'
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


def Remap_phase(Z_mod, g1_ϕ, g2_ϕ):
    Z_mod1 = copy.copy(Z_mod)
    Z_mod2 = copy.copy(Z_mod)
    for i1 in range(np.shape(Z_mod)[0]):

        Z_mod1[i1, :] = g1_ϕ(Z_mod[i1, :])
        Z_mod2[i1, :] = g2_ϕ(Z_mod[i1, :])
    return (Z_mod1, Z_mod2)


###############################################################################
# Calculate phase map
###############################################################################

# Phase mapping details (ϕ)
(ϕ_A, ϕ_B, ϕ_g) = Fit_phase()
ϕ_min = 0
ϕ_max = max(ϕ_g)
ϕ_rng = (ϕ_min, ϕ_max)
g1_ϕ = interp1d(np.linspace(ϕ_min, ϕ_max, 255), range(255))
g2_ϕ = interp1d(ϕ_g, range(255))

###############################################################################
# Specify
###############################################################################

# LCOS size (# pixels in x & y)
LCOS_δy = 70
LCOS_δx = 80

# Subhologram size and location
Hol_δx = 40
Hol_δy = 40
Hol_cx = int(LCOS_δx / 2)
Hol_cy = int(LCOS_δy / 2)

# Phase (ϕ) upper and lower limits
ϕ_uplim = ϕ_max - 0.4
ϕ_lwlim = 0.4

# Grating metrics (specify Λ (period) and φ (rotation angle))
Λ = 5
φ = 2 * np.pi / 4

###############################################################################
# Construct some stuff for the code
###############################################################################

LCOS_δyx = (LCOS_δy, LCOS_δx)
Hol_δyx = (Hol_δy, Hol_δx)
Hol_cyx = (Hol_cy, Hol_cx)
ϕ_lims = (ϕ_lwlim, ϕ_uplim)

# Define holo params
Holo_params = (Λ, φ, *Hol_δyx, *φ_lims)

# Calculate sub hologram (Holo_s)
Holo_s = Holo_tilt(*Holo_params)
Z = Holo_s[0]
Z_mod = Holo_s[1]

# Remap phase with non linear ϕ map
(Z_mod1, Z_mod2) = Remap_phase(Z_mod, g1_ϕ, g2_ϕ)

# Calculate full hologram (Holo_f)
Holo_f0 = Add_Holo(*Hol_cyx, Z_mod, *LCOS_δyx)
Holo_f1 = Add_Holo(*Hol_cyx, Z_mod1, *LCOS_δyx)
Holo_f2 = Add_Holo(*Hol_cyx, Z_mod2, *LCOS_δyx)

###############################################################################
# Plotting
###############################################################################


# pl.figure('Sub hologram')
# im1 = plt.imshow(Z_mod, cmap)
# plt.imshow
# plt.savefig('holo1.png', transparent=True)

# pl.figure('g vs x')
# pl.plot(range(Hol_δy), g2_ϕ(Z_mod[:, 0]), ':b.')
# pl.plot(range(Hol_δy), g1_ϕ(Z_mod[:, 0]), ':r.')
# pl.title('g vs x position')

# pl.figure('g vs ϕ')
# pl.plot(g2_ϕ(Z_mod[:, 0]), ':b.')
# pl.plot(g1_ϕ(Z_mod[:, 0]), ':r.')
# pl.title('g vs ϕ position')

# pl.figure('SH profile')
# pl.plot(range(Hol_δy), Z_mod1[:, 0], ':b.')
# pl.plot(range(Hol_δy), Z_mod2[:, 0], ':r.')
# pl.title('g vs x position')

# pl.figure('Full Phase')
# im0 = plt.imshow(Holo_f0, cmap)
# pl.title('Λ = %s, ϕ = %sπ' % (Λ, φ / np.pi))
# plt.colorbar()
# plt.clim(ϕ_min, ϕ_max)
# print(np.max(Holo_f0))

pl.figure('Full Hologram1')
im0 = plt.imshow(Holo_f1, cmap)
pl.title('Λ = %s, ϕ = %sπ' % (Λ, φ / np.pi))
plt.colorbar()
plt.clim(0, 255)
plt.ylabel('LCOS y axis')
plt.xlabel('LCOS x axis')

pl.figure('Full Hologram2')
im1 = plt.imshow(Holo_f2, cmap)
pl.title('Λ = %s, ϕ = %sπ' % (Λ, φ / np.pi))
plt.colorbar()
plt.clim(0, 255)
plt.ylabel('LCOS y axis')
plt.xlabel('LCOS x axis')


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

pl.figure('Hologram Profiles')
plt.plot(range(LCOS_δy), g1_ϕ(ϕ_lwlim) *
         np.ones(np.shape(Holo_f1[:, Hol_cx])), 'xkcd:blue')
plt.plot(range(LCOS_δy), g1_ϕ(ϕ_uplim) *
         np.ones(np.shape(Holo_f1[:, Hol_cx])), 'xkcd:blue')

plt.plot(range(LCOS_δy), Holo_f1[:, Hol_cx], '.:',
         color='xkcd:light blue', mfc='xkcd:dark blue')

plt.plot(range(LCOS_δy), g2_ϕ(ϕ_lwlim) *
         np.ones(np.shape(Holo_f2[:, Hol_cx])), 'xkcd:red')
plt.plot(range(LCOS_δy), g2_ϕ(ϕ_uplim) *
         np.ones(np.shape(Holo_f2[:, Hol_cx])), 'xkcd:red')
plt.plot(range(LCOS_δy), Holo_f2[:, Hol_cx], '.:',
         color='xkcd:light red', mfc='xkcd:dark red')

pl.title('Λ = %s, ϕ = %sπ' % (Λ, φ / np.pi))
plt.ylim([0, 255])
plt.ylabel('Greyscale value [0:255]')
plt.xlabel('LCOS y axis')
# plt.grid()


pl.figure('ϕ mapping')
pl.plot(range(255), ϕ_g, color='xkcd:red', label='From φ-mapping')
pl.plot(g2_ϕ(ϕ_lwlim) * np.ones(100),
        np.linspace(ϕ_min, ϕ_lwlim, 100), ':', color='xkcd:light red')
pl.plot(g2_ϕ(ϕ_uplim) * np.ones(100),
        np.linspace(ϕ_min, ϕ_uplim, 100), ':', color='xkcd:light red')

pl.plot(range(255), np.linspace(ϕ_min, ϕ_max, 255),
        color='xkcd:blue', label='Linear mapping')
pl.plot(g1_ϕ(ϕ_lwlim) * np.ones(100),
        np.linspace(ϕ_min, ϕ_lwlim, 100), ':', color='xkcd:light blue')
pl.plot(g1_ϕ(ϕ_uplim) * np.ones(100),
        np.linspace(ϕ_min, ϕ_uplim, 100), ':', color='xkcd:light blue')
pl.plot(range(255), ϕ_uplim * np.ones(255), color='xkcd:black')
pl.plot(range(255), ϕ_lwlim * np.ones(255), color='xkcd:black')

plt.legend()
plt.ylabel('Phase (φ)')
plt.xlabel('Grayscale value [0:255]')

plt.show()
