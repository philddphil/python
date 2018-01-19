#  Replay2D.py
#
#  Code to calculate 2D replay field using psf.
#
#  ROADMap Systems Ltd
#
#  Brian Robertson
#
#  23/Feb/2017
#
#  Version 1 - Basic code layout (23/Feb/2016) - based on previous Python code
#              "4pi_grating_polynomial_passband.py".
#  Version 2 - Adapted to plot replay field of a linear phase mask (phase split
#              or linear Axicon (01/March/2017).
#  Version 3 - Update to calculate full replay field (14/03/2017).
#
#  Version 4 - Update to add mode-overlap calculation.
#
#  Version 5 - Tidied up code, and identified bugs (19/08/2017).  Renamed code
#              as Replay2D.py

#  Tasks
#
#  1) Convert to python version 3.6
#  2) Correct for initial grating definition (px,py).  Read HWU paper.
#  3) Check mode-overlap integral.
#  4) psf definition - read Swedish paper.
#  5) psf definition - read CAPE paper.
#  6) Convert into functions.
#  7) Convert into classes.
#  8) Add sin grating compensation algorithm.
#  9) Put back Axicon functionality.
#
# ****************************** HOUSEKEEPING ****************************
from numpy import*
from pylab import *
from math import*
import sys
import time

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

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


def file_save():
    f = tkFileDialog.asksaveasfile(mode='w', defaultextension=".txt")
    if f is None:
        # asksaveasfile return `None` if dialog closed with "cancel".
        return
    text2save = str(text.get(1.0, END))  # starts from `1.0`, not `0.0`
    f.write(text2save)
    f.close()  # `()` was missing.

λ = 1550e-9
c = 3e8
N_pixels = 50
pixel = 6.4e-6
w = N_pixels * pixel / 5.0
wf = 5.2e-6
f = wf * pi * w / λ
zp = f
p = 10
p2 = p / 3
theta_m = np.arcsin(λ / (p * pixel))
theta_m_2 = np.arcsin(λ / (p2 * pixel))
yp = f * tan(theta_m)
xp = f * tan(theta_m)
yp2 = f * tan(theta_m_2)
xp2 = f * tan(theta_m_2)

# Definition of propagation vector
v_mag = np.sqrt(f**2 + xp**2 + yp**2)
nx = xp / v_mag
ny = yp / v_mag
nz = zp / v_mag

v_mag2 = np.sqrt(f**2 + xp2**2 + yp2**2)
nx2 = xp2 / v_mag
ny2 = yp2 / v_mag
nz2 = zp / v_mag

# Number of phase levels
levels = 128.0

# ***************************** DEFINE FUNCTIONS **************************

# Function 1 - generation of base hologram
# Function 2 - add psf of pixels of SLM
# Function 3 - define Gaussian intensity profile (2D)
# Function 4 - replay field calculation.
# Function 5 - CEF calculation.

# ***************************** FILE INFORMATION **************************

print('')
print('Program to analyse replay field of sub-hologram')
print('')

print('start calculation')
print(' ')
start = time.time()


##############################################################################
# 1 - Generate Hologram
##############################################################################

(_, phase_SLM) = prd.holo_tilt(8, np.pi / 2, N_pixels, N_pixels)

phase_SLM = phase_SLM[:, 0]

# fig1 = plt.figure('fig1')
# ax1 = fig1.add_subplot(1, 1, 1)
# fig1.patch.set_facecolor(cs['mdk_dgrey'])
# plt.plot(phase_SLM)
# xlabel('x label')
# ylabel('y label')
# title('title')
# Increase resolution

# Increase calculation resolution by representing each pixel by NxN
# elements to give a total phase and amplitude field of NTxNT points

res = 8
NT = N_pixels * res

Np = zeros([NT])
Amp_Np = zeros([NT])

# Set central phase value in each pixel
for ii in range(NT):
    if ((ii / res) == floor(ii / res)):
        Np[int(ii - res + 1 + res / 2)] = phase_SLM[int(ii / res)]

padded_phase_SLM = Np


##############################################################################
# 2 - Generate PSF
##############################################################################
# Calculate phase profile for NTxNT points by convolving Np with the psf
# phase_SLM = conv2(Np,pixel_define);

# Define a point-spread function representing each pixel
psf = zeros([NT])

NN = 2
w_edge = 5e-6

for ii in range(NT):
    xpos = ((-NT / 2) + 0.5 + ii) * pixel / res
    psf[ii] = 1 * np.exp(-(xpos)**2 / (2 * (w_edge)**2))**NN
    # psf[ii,jj] = 0;
    # if (abs(xpos) < pixel/2) and (abs(ypos) < pixel/2):
    # psf[ii,jj] = 1;

max_psf = np.max(np.max(psf))
psf = psf / max_psf

# fig2 = plt.figure('fig2')
# ax21 = fig1.add_subplot(1, 1, 1)
# fig2.patch.set_facecolor(cs['mdk_dgrey'])
# plt.plot(psf)
# xlabel('x label')
# ylabel('y label')
# title('title')

##############################################################################
# 3 - Define Gaussian intensity profile (2D)
##############################################################################
SLM_amplt = zeros([NT])

for ii in range(NT):
    xSLM = (-NT / 2 + 0.5 + ii) * pixel / res
    amplitude = np.exp(-1 * (xSLM**2) / (w**2))
    SLM_amplt[ii] = amplitude

# Define new phase profile
phase_SLM_1 = fftshift(fft(fftshift(padded_phase_SLM))) * \
    fftshift(fft(fftshift(psf)))
phase_SLM_2 = fftshift(ifft(fftshift(phase_SLM_1)))
phase_SLM_2 = 2 * np.pi * phase_SLM_2 / np.max(np.abs(phase_SLM_2))
print('max ϕ =', np.max(np.max(np.abs(phase_SLM_2))))

# fig3 = plt.figure('fig3')
# fig3.patch.set_facecolor(cs['mdk_dgrey'])
# plt.plot(phase_SLM_2)
# xlabel('x label')
# ylabel('y label')
# title('title')

##############################################################################
# 4 - Calculate replay field
##############################################################################

# Define phase distribution when there is no hologram displayed
SLM_zero = zeros([NT])

# Define zero padding factor, pad, and generate associated replay field
# calaculation matrices
pad = 6
E_calc = zeros([pad * NT])
E_calc_phase = zeros([pad * NT]) * 0j
E_calc_amplt = zeros([pad * NT]) * 0j

# Calculation of replay field when no grating is displayed ###################
E_calc_phase[0:NT] = SLM_zero[:]
E_calc_amplt[0:NT] = SLM_amplt[:]
E_replay_zero = fftshift(
    fft(fftshift(E_calc_amplt * np.exp(1j * E_calc_phase))))
I_replay_zero = (abs(E_replay_zero))**2
# Maximum intensity
I_max_zero = np.max(np.max(I_replay_zero))
# Normalized replay field
I_replay_zero = I_replay_zero / I_max_zero

# Calculation of replay field when grating is displayed ######################
E_calc_phase[0:NT] = phase_SLM_2[:]
E_calc_amplt[0:NT] = SLM_amplt[:]
E_replay = fftshift(fft(fftshift(E_calc_amplt * np.exp(1j * E_calc_phase))))
I_replay = (abs(E_replay))**2
# Maximum intensity
I_max_signal = np.max(np.max(I_replay))
# Replay intensity distribution normalized with respect to the undiffracted
# zeroth order
I_replay = I_replay / I_max_zero

# Corresponding insertion loss
Loss = I_max_signal / I_max_zero
print('Loss = ', Loss)

I1_final = zeros([200])
I1_final = 10 * np.log10(I_replay_zero[int(NT * pad / 2 - 100):
                                       int(NT * pad / 2 + 100)])
I1_final[I1_final < -60] = -60

I2_final = zeros([200])
I2_final = 10 * np.log10(I_replay[int(NT * pad / 2 - 100):
                                  int(NT * pad / 2 + 100)])
I2_final[I2_final < -60] = -60


fig5 = figure('fig5')
fig5.patch.set_facecolor(cs['mdk_dgrey'])

ax51 = fig5.add_subplot(221)
plt.plot(I1_final)

ax52 = fig5.add_subplot(222)
plt.plot(I2_final)

ax53 = fig5.add_subplot(223)
plt.plot(np.abs(E_calc_phase))

ax54 = fig5.add_subplot(224)
plt.plot(np.abs(E_calc_amplt))


show()
