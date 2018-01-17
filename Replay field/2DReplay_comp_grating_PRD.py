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
from scipy import signal
import sys
import time
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.patches as patches

##############################################################################
# Import some extra special libraries from my own repo
##############################################################################
sys.path.insert(0, r'C:\Users\Philip\Documents\Python\Local Repo\library')
import useful_defs_prd as prd
cs = prd.palette()

##############################################################################
# Define values
##############################################################################
π = np.pi
δx = 50
δy = 50
w = 30
Λ = 6
ϕ = π / 4
px_edge = 4  # blurring of rect function - bigger = blurier
px_pad = 8
fft_pad = 4


##############################################################################
# CODE STRUCTURE
##############################################################################
# Function 1 - generation of base hologram
# Function 2 - define Gaussian intensity profile (2D)
# Function 3 - add psf of pixels of SLM
#            - plots (i)
# Function 4 - replay field calculation.
# Function 5 - CEF calculation.
#            - plots (ii)

##############################################################################
# 1 - Generate Hologram
##############################################################################
(_, phase_SLM) = prd.holo_tilt(Λ, ϕ, δx, δy)

# Increase resolution
# Increase calculation resolution by representing each pixel by px_pad**2
# elements to give a total phase and amplitude field of LC x LC points

LCx = np.shape(phase_SLM)[0] * (2 * px_pad + 1)
LCy = np.shape(phase_SLM)[1] * (2 * px_pad + 1)
LC_field = prd.Pad_A_elements(phase_SLM, px_pad)

SLM_x = range(LCx)
SLM_y = range(LCy)
coords = np.meshgrid(SLM_x, SLM_y)


##############################################################################
# 2 - Define Gaussian intensity profile (2D)
##############################################################################
G1 = prd.Gaussian_2D(coords, 1, LCx / 2, LCy / 2,
                     0.25 * w * (2 * px_pad + 1),
                     0.25 * w * (2 * px_pad + 1))
E_field = np.reshape(G1, (LCx, LCy))
(LC_cx, LC_cy) = prd.max_i_2d(E_field)


##############################################################################
# 3 - Generate PSF
##############################################################################
# Calculate phase profile for NTxNT points by convolving Np with the psf
# phase_SLM = conv2(Np,pixel_define);
# Define a point-spread function representing each pixel
R0 = zeros((LCx, LCy))
R0[LC_cx - px_pad:LC_cx + px_pad + 1,
    LC_cy - px_pad:LC_cy + px_pad + 1] = 1
R0 = prd.n_G_blurs(R0, 1, px_edge)

# Define new phase profile
phase_SLM_1 = fftshift(fft2(fftshift(LC_field))) * \
    fftshift(fft2(fftshift(R0)))
phase_SLM_2 = fftshift(ifft2(fftshift(phase_SLM_1)))


##############################################################################
# aside - plots (i)
##############################################################################

fig3 = figure('fig3')
fig3.patch.set_facecolor(cs['mdk_dgrey'])

ax3_1 = fig3.add_subplot(411)
plt.plot(phase_SLM[0, 0: 2 * Λ], 'o')
plt.plot(np.linspace(0, 2 * Λ, 2 * Λ * (2 * px_pad + 1)) - 0.5,
         phase_SLM_2[px_pad, 0: 2 * Λ * (2 * px_pad + 1)])

ax3_3 = fig3.add_subplot(423)
plt.plot(R0[LC_cy, LC_cx - 2 * px_pad - 1:LC_cx + 2 * px_pad + 2], 'o-')
plt.plot([px_pad + 1, px_pad + 1], [1, 0], c=cs['ggblue'])
plt.plot([3 * px_pad + 1, 3 * px_pad + 1], [1, 0], c=cs['ggblue'])

ax3_4 = fig3.add_subplot(424)
plt.imshow(R0[LC_cx - 2 * px_pad:LC_cx + 2 * px_pad + 1,
              LC_cy - 2 * px_pad:LC_cy + 2 * px_pad + 1])
ax3_4.add_patch(
    patches.Rectangle((px_pad, px_pad),
                      2 * px_pad, 2 * px_pad,
                      fill=False, edgecolor=cs['ggred'], lw=2))

ax3_5 = fig3.add_subplot(425)
plt.imshow(LC_field[0:Λ * 2 * px_pad + 1,
                    0:Λ * 2 * px_pad + 1])

ax3_6 = fig3.add_subplot(426)
plt.imshow(abs(phase_SLM_2[0:Λ * 2 * px_pad + 1,
                           0:Λ * 2 * px_pad + 1]))

ax3_7 = fig3.add_subplot(414)
plt.plot(np.linspace(0, δx, LCx), E_field[LC_cx, :])

plt.tight_layout()


##############################################################################
# 4 - Calculate replay field
##############################################################################
# Define phase distribution when there is no hologram displayed
SLM_zero = zeros([LCx, LCy])

# Define zero padding factor, pad, and generate associated replay field
# calaculation matrices
E_calc = zeros([fft_pad * LCx, fft_pad * LCy])
E_calc_phase = zeros([fft_pad * LCx, fft_pad * LCy]) * 0j
E_calc_amplt = E_calc_phase

# Calculation of replay field when no grating is displayed ###################
E_calc_phase[0:LCx, 0:LCy] = SLM_zero[:, :]
E_calc_amplt[0:LCx, 0:LCy] = E_field[:, :]
E_replay_zero = fftshift(
    fft2(fftshift(E_calc_amplt * np.exp(1j * E_calc_phase))))
I_replay_zero = (abs(E_replay_zero))**2

# Maximum intensity
I_max_zero = np.max(np.max(I_replay_zero))

# Normalized replay field
I_replay_zero = I_replay_zero / I_max_zero
E_calc_phase = E_calc * 0j

# Calculation of replay field when grating is displayed ######################
E_calc_phase[0:LCx, 0:LCy] = phase_SLM_2[:, :]
E_calc_amplt[0:LCx, 0:LCy] = E_field[:, :]
E_replay = fftshift(fft2(fftshift(E_calc_amplt * np.exp(1j * E_calc_phase))))
I_replay = (abs(E_replay))**2
# Maximum intensity
I_max_signal = np.max(np.max(I_replay))
# Replay intensity distribution normalized with respect to the undiffracted
# zeroth order
I_replay = I_replay / I_max_zero

# Corresponding insertion loss
Loss = I_max_signal / I_max_zero
print('Loss = ', Loss)

I1_final = zeros([200, 200])
I1_final = 10 * np.log10(I_replay_zero[int(LCx * fft_pad / 2 - 100):
                                       int(LCx * fft_pad / 2 + 100),
                                       int(LCy * fft_pad / 2 - 100):
                                       int(LCy * fft_pad / 2 + 100)])
I1_final[I1_final < -60] = -60

I2_final = zeros([200, 200])
I2_final = 10 * np.log10(I_replay[int(LCx * fft_pad / 2 - 100):
                                  int(LCx * fft_pad / 2 + 100),
                                  int(LCy * fft_pad / 2 - 100):
                                  int(LCy * fft_pad / 2 + 100)])
I2_final[I2_final < -60] = -60

##############################################################################
# aside - plots (ii)
##############################################################################
# fig3 = plt.figure('fig3')
# fig3.patch.set_facecolor(cs['mdk_dgrey'])
# plt.imshow(I_replay)
# xlabel('Pixel number')
# ylabel('Phase delay (radians')
# title('Optimized ideal phase profile (x-axis slice)')

# fig3 = plt.figure('fig3')
# fig3.patch.set_facecolor(cs['mdk_dgrey'])
# plt.imshow(np.abs(phase_SLM_2))
# xlabel('N_pixels*resolution')
# ylabel('N_pixels*resolution')
# plt.colorbar()
# title('Optimized sub-hologram profile (ideal)')

fig5 = figure('fig5')
fig5.patch.set_facecolor(cs['mdk_dgrey'])
ax51 = fig5.add_subplot(221)
imshow(I1_final)

ax52 = fig5.add_subplot(222)
imshow(I2_final)
plt.colorbar()

ax53 = fig5.add_subplot(223)
imshow(np.abs(E_calc_phase))
plt.colorbar()
plt.title('E_calc_phase', fontsize=8)

ax54 = fig5.add_subplot(224)
imshow(np.abs(E_calc_amplt))
plt.colorbar()
ax54.set_title('E_calc_amplt', fontsize=8)

plt.tight_layout()

fig1 = figure('fig1')
fig1.patch.set_facecolor(cs['mdk_dgrey'])
ax1 = fig1.add_subplot(111)
ax1.set_xlabel('x axis')
ax1.set_ylabel('y axis')

plt.imshow(I2_final)

fig2 = figure('fig2')
fig2.patch.set_facecolor(cs['mdk_dgrey'])
ax2 = fig2.add_subplot(111)
ax2.set_xlabel('x axis')
ax2.set_ylabel('y axis')

plt.plot(psf[LC_cy, LC_cx - 2 * px_pad - 1:LC_cx + 2 * px_pad + 2], 'o-')
plt.plot([px_pad + 1, px_pad + 1], [1, 0], c=cs['ggblue'])
plt.plot([3 * px_pad + 1, 3 * px_pad + 1], [1, 0], c=cs['ggblue'])


fig4 = figure('fig4')
fig4.patch.set_facecolor(cs['mdk_dgrey'])
ax4 = fig4.add_subplot(111)
ax4.set_xlabel('x axis')
ax4.set_ylabel('y axis')
plt.plot(phase_SLM[0, 0: 2 * Λ], 'o')
plt.plot(np.linspace(0, 2 * Λ, 2 * Λ * (2 * px_pad + 1)) - 0.5,
         phase_SLM_2[px_pad, 0: 2 * Λ * (2 * px_pad + 1)])

# fig6 = plt.figure('fig6')
# ax6 = fig6.gca(projection='3d')
# fig6.patch.set_facecolor(cs['mdk_dgrey'])
# ax6.w_xaxis.set_pane_color(cs['mdk_dgrey'])
# ax6.w_yaxis.set_pane_color(cs['mdk_dgrey'])
# ax6.w_zaxis.set_pane_color(cs['mdk_dgrey'])
# XX = np.arange(1, 201, 1)
# YY = np.arange(1, 201, 1)
# XX, YY = np.meshgrid(XX, YY)

# Make data.
# Plot the surface.
# surf = ax6.plot_surface(XX, YY, I2_final, cmap=cm.coolwarm,
#                         linewidth=0, antialiased=False)

show()

prd.PPT_save_2d(fig1, ax1, 'plot0.png')
prd.PPT_save_2d(fig2, ax2, 'plot1.png')
prd.PPT_save_2d(fig4, ax4, 'plot2.png')
