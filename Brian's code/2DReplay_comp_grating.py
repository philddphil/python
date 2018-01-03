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
#******************************* HOUSEKEEPING ****************************
from numpy import*
from pylab import *
from math import*
import sys
from scipy.optimize import minimize
from scipy.special import erf
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm


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
    if f is None:  # asksaveasfile return `None` if dialog closed with "cancel".
        return
    text2save = str(text.get(1.0, END))  # starts from `1.0`, not `0.0`
    f.write(text2save)
    f.close()  # `()` was missing.

#***************************** SET UP PARAMETERS *************************

wavelength = 1550e-9
clight = 3e8
N_pixels = 50
pixel = 6.4e-6
w = N_pixels * pixel / 5.0
wf = 5.2e-6
f = wf * pi * w / wavelength
zp = f
p = 10.0
p2 = p / 2
theta_m = np.arcsin(wavelength / (p * pixel))
theta_m_2 = np.arcsin(wavelength / (p2 * pixel))
yp = f * tan(theta_m)
xp = f * tan(theta_m)
yp2 = f * tan(theta_m_2)
xp2 = f * tan(theta_m_2)

# Define axicon wedge angle
beta = 0 * pi / 180.0
beta2 = -1 * beta

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

#***************************** DEFINE FUNCTIONS **************************

# Function 1 - generation of base hologram
# Function 2 - add psf
# Function 3 - define Gaussian intensity profile (2D)
# Function 4 - replay field calculation.
# Function 5 - CEF calculation.

#***************************** FILE INFORMATION **************************

# File information
print('')
print('Program to analyse replay field of sub-hologram')
print('')

print('start calculation')
print(' ')
start = time.time()

#********************************** MAIN LOOP ****************************

##############################################################################
# 1 - Generate Hologram
##############################################################################
shift = 0.6
weight = 0.2

phase_SLM = zeros([N_pixels, N_pixels])
grey_SLM = zeros([N_pixels, N_pixels])

for n in range(N_pixels):
    for m in range(N_pixels):

        xSLM = n * pixel
        ySLM = m * pixel
        h = (nx * xSLM + ny * ySLM) / nz
        h2 = (nx2 * xSLM + ny2 * ySLM) / nz2
        # Corresponding phase step
        phase = (2 * pi * (h / wavelength))
        # Phase step according to modulo 2pi algorithm
        diff_phase = 2 * pi * ((phase / (2 * pi)) - floor(phase / (2 * pi)))

        p1c = weight * \
            np.sin(2 * pi * ((xSLM / (p2 * pixel) + ySLM / (p2 * pixel)) + shift))
        diff_phase = diff_phase + p1c

        if diff_phase > ((levels - 1 + 0.5) / levels) * 2 * pi:
            diff_phase = 0
        # Quantize phase according to number of available phase levels
        q1 = floor((diff_phase / (2 * pi) * levels))
        quant_phase = q1 * 2 * pi / levels
        # Convert to bit-map value (scaled between 0 and 255)
        q2 = round(255 * q1 / levels)
        # Output arrays for phase delay and grey level
        phase_SLM[m, n] = quant_phase
        grey_SLM[m, n] = q2

fig1 = plt.figure('fig1')
ax1 = fig1.add_subplot(1, 1, 1)
fig1.patch.set_facecolor(cs['mdk_dgrey'])
plt.plot(phase_SLM[1, :], '.:')
ax1.set_xlabel('Pixel number')
ax1.set_ylabel('Phase delay (radians')
title('Optimized ideal phase profile (x-axis slice)')


# Increase resolution

# Increase calculation resolution by representing each pixel by NxN
# elements to give a total phase and amplitude field of NTxNT points

# Set central phase value in each pixel
res = 8
NT = N_pixels * res

Np = zeros([NT, NT])
Amp_Np = zeros([NT, NT])

# Set central phase value in each pixel
for ii in range(NT):
    for jj in range(NT):
        if ((ii / res) == floor(ii / res)) and ((jj / res) == floor(jj / res)):
            Np[int(ii - res + 1 + res / 2),
               int(jj - res + 1 + res / 2)] = phase_SLM[int(ii / res),
                                                        int(jj / res)]

phase_SLM = Np
print(np.shape(phase_SLM),np.shape(Np))

##############################################################################
# 2 - Generate PSF
##############################################################################
# Calculate phase profile for NTxNT points by convolving Np with the psf
# phase_SLM = conv2(Np,pixel_define);

# Define a point-spread function representing each pixel
psf = zeros([NT, NT])

NN = 2
w_edge = 1e-6

for ii in range(NT):
    for jj in range(NT):
        xpos = ((-NT / 2) + 0.5 + ii) * pixel / res
        ypos = ((-NT / 2) + 0.5 + jj) * pixel / res
        rpos = np.sqrt(xpos**2 + ypos**2)
        psf[ii, jj] = 1 * np.exp(-(rpos)**2 / (2 * (w_edge)**2))**NN
        # psf[ii,jj] = 0;
        # if (abs(xpos) < pixel/2) and (abs(ypos) < pixel/2):
        # psf[ii,jj] = 1;

max_psf = np.max(np.max(psf))
psf = psf / max_psf / 7.0

# Define beam amplitude profile

SLM_amplt = zeros([NT, NT])

for ii in range(NT):
    for jj in range(NT):
        xSLM = (-NT / 2 + 0.5 + ii) * pixel / res
        ySLM = (-NT / 2 + 0.5 + jj) * pixel / res
        amplitude = np.exp(-1 * (xSLM**2 + ySLM**2) / (w**2))
        SLM_amplt[ii, jj] = amplitude


# Define new phase profile
phase_SLM_1 = fftshift(fft2(fftshift(phase_SLM))) * \
    fftshift(fft2(fftshift(psf)))
phase_SLM_2 = fftshift(ifft2(fftshift(phase_SLM_1)))

max_phase = np.max(np.max(phase_SLM_2))
print('Max phase', max_phase)
min_phase = np.min(np.min(phase_SLM_2))
print('Min phase', min_phase)
# phase_SLM_2 = phase_SLM_2*(2*pi/(max_phase));




# Lock
phase_SLM_L = zeros([NT, NT])

for n in range(NT):
    for m in range(NT):
        # Calculate wavefront ramp, h, at location (xp, yp)

        # Calculate conical phase profile
        xA = (-NT / 2 + 0.5 + n - 1) * pixel / res
        yA = (-NT / 2 + 0.5 + m - 1) * pixel / res
        radius = np.sqrt(xA**2 + yA**2)
        h1 = radius * tan(beta2)
        # Corresponding phase step
        phase_L = (2 * pi * (h1 / wavelength))
        # Phase step according to modulo 2pi algorithm
        diff_phase_L = 2 * pi * \
            ((phase_L / (2 * pi)) - floor(phase_L / (2 * pi)))
        # Quantize phase according to number of available levels
        q1_L = floor((diff_phase_L / (2 * pi)) * levels)
        quant_phase_L = q1_L * 2 * pi / levels

        # Output array for phase delay
        phase_SLM_L[n, m] = quant_phase_L

# Main calculation

# Define phase distribution when there is no hologram displayed
SLM_zero = zeros([NT, NT])

# Define zero padding factor, pad, and generate associated replay field
# calaculation matrices
pad = 4
E_calc = zeros([pad * NT, pad * NT])
E_calc_phase = zeros([pad * NT, pad * NT]) * 0j
E_calc_amplt = zeros([pad * NT, pad * NT]) * 0j

# Calculation of replay field when no grating is displayed
E_calc_phase[0:NT, 0:NT] = SLM_zero[:, :]
E_calc_amplt[0:NT, 0:NT] = SLM_amplt[:, :]
E_replay_zero = fftshift(
    fft2(fftshift(E_calc_amplt * np.exp(1j * E_calc_phase))))
I_replay_zero = (abs(E_replay_zero))**2
# Maximum intensity
I_max_zero = np.max(np.max(I_replay_zero))
# Normalized replay field
I_replay_zero = I_replay_zero / I_max_zero


# Calculation of replay field when grating is displayed
E_calc_phase[0:NT, 0:NT] = phase_SLM_2[:, :] + phase_SLM_L[:, :]
E_calc_amplt[0:NT, 0:NT] = SLM_amplt[:, :]
E_replay = fftshift(fft2(fftshift(E_calc_amplt * np.exp(1j * E_calc_phase))))
I_replay = (abs(E_replay))**2
# Maximum intensity
I_max_signal = np.max(np.max(I_replay))
# Replay intensity distribution normalized with respect to the undiffracted
# zeroth order
I_replay = I_replay / I_max_zero

# Corresponding insertion loss
Loss = I_max_signal / I_max_zero
print(Loss)

# Calculate scaling vector and plot
x_vec = zeros([NT * pad])

# Calculate replay field scaled distance (in microns)
for ii in range(NT * pad):
    pos = ((1 + NT * pad / 2) - ii) * f * \
        wavelength / ((pixel / res) * pad * NT)
    x_vec[ii] = pos * 1e6

# Calculate Mode-overlap integral

range = 200

[xcen, ycen] = np.where(I_replay_zero == I_replay_zero.max())
print('Central coordinates', xcen, ycen)
Replay_zero = abs(
    E_replay_zero[int(xcen - range):int(xcen + range),
                  int(ycen - range):int(ycen + range)])

[xpos, ypos] = np.where(I_replay == I_replay.max())
print('Signal position', xpos, ypos)
Replay_sig = abs(E_replay[int(xpos - range):
                          int(xpos + range),
                          int(ypos - range):
                          int(ypos + range)])

xxt = xcen - (xpos - xcen)
yxt = ycen - (ypos - ycen)
print('-1 order crosstalk position', xxt, yxt)
Crosstalk_sig = abs(E_replay[int(xxt - range):
                             int(xxt + range),
                             int(yxt - range):
                             int(yxt + range)])

points = 2 * range


# Alternative calculation
test = (np.sum(((Replay_sig) * (Replay_zero))))**2 / \
    ((np.sum(np.abs(E_replay)**2)) * (np.sum(np.abs(E_replay_zero)**2)))
print('test signal = ', test)

test = (np.sum(((Crosstalk_sig) * (Replay_zero))))**2 / \
    ((np.sum(np.abs(E_replay)**2)) * (np.sum(np.abs(E_replay_zero)**2)))
print('test crosstalk = ', 10 * log10(test))

I1_final = zeros([200, 200])
I1_final = 10 * np.log10(I_replay_zero[int(NT * pad / 2 - 100):
                                       int(NT * pad / 2 + 100),
                                       int(NT * pad / 2 - 100):
                                       int(NT * pad / 2 + 100)])
I1_final[I1_final < -60] = -60

I2_final = zeros([200, 200])
I2_final = 10 * np.log10(I_replay[int(NT * pad / 2 - 100):
                                  int(NT * pad / 2 + 100),
                                  int(NT * pad / 2 - 100):
                                  int(NT * pad / 2 + 100)])
I2_final[I2_final < -60] = -60

end = time.time()

fig3 = plt.figure('fig3')
fig3.patch.set_facecolor(cs['mdk_dgrey'])
plt.imshow(phase_SLM)
xlabel('Pixel number')
ylabel('Phase delay (radians')
title('Optimized ideal phase profile (x-axis slice)')



fig2 = plt.figure('fig2')
fig2.patch.set_facecolor(cs['mdk_dgrey'])
plt.imshow(Np)
xlabel('N_pixels*resolution')
ylabel('N_pixels*resolution')
title('Optimized sub-hologram profile (ideal)')
show()

# fig4 = figure('fig4')
# ax4 = fig4.add_subplot(1, 1, 1)
# fig4.patch.set_facecolor(cs['mdk_dgrey'])
# plt.subplot(221)
# imshow(I1_final, cmap='gray')
# plt.colorbar()
# plt.title('Undiffracted beam')
# plt.subplot(222)
# imshow(I2_final, cmap='gray')
# plt.colorbar()
# plt.title('Diffracted beam')

# fig5 = figure('fig5')
# ax5 = fig5.add_subplot(1, 1, 1)
# fig5.patch.set_facecolor(cs['mdk_dgrey'])
# plt.subplot(221)
# imshow(I1_final)
# plt.title('Undiffracted beam')
# plt.colorbar()
# plt.subplot(222)
# imshow(I2_final)
# plt.colorbar()
# plt.title('Diffracted beam')


# fig6 = plt.figure('fig6')
# ax6 = fig6.gca(projection='3d')
# fig6.patch.set_facecolor(cs['mdk_dgrey'])
# ax6.w_xaxis.set_pane_color(cs['mdk_dgrey'])
# ax6.w_yaxis.set_pane_color(cs['mdk_dgrey'])
# ax6.w_zaxis.set_pane_color(cs['mdk_dgrey'])
# XX = np.arange(1, 201, 1)
# YY = np.arange(1, 201, 1)
# XX, YY = np.meshgrid(XX, YY)

# # Make data.
# # Plot the surface.
# surf = ax6.plot_surface(XX, YY, I2_final, cmap=cm.coolwarm,
#                         linewidth=0, antialiased=False)

show()
