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
import numpy as np
import sys
import copy
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.interpolate import interp1d


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
px_edge = 0  # blurring of rect function - bigger = blurier
px_pad = 8
fft_pad = 4
px = 6.4e-6
λ = 1.55e-6
f = 9.1e-3
w = 50

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
f0 = (r"C:\Users\Philip\Documents\Technical Stuff\Hologram optimisation"
      r"\Algorithmic implementation\180227\Post realignment\Port 2\fibre2.csv")

p0 = (r"C:\Users\Philip\Documents\Technical Stuff\Hologram optimisation"
      r"\Replay field calculation")
p1 = (r"C:\Users\Philip\Documents\Technical Stuff\Hologram optimisation"
      r"\Replay field calculation\180302")

f1 = p0 + r'\Phase Ps.csv'
f2 = p0 + r'\Phase greys.csv'
holo_data = np.genfromtxt(f0, delimiter=',')

Λ = holo_data[0]
φ = (np.pi / 180) * 90
H_δx = int(holo_data[4])
H_δy = int(holo_data[5])
H_δx = 80
H_δy = 80

ϕ_lw = π * 0.5
ϕ_up = π * 2.5

os_lw = π * -0.1
os_up = π * -0.1
osw_lw = 0
osw_up = 0

off = holo_data[16]

Holo_params = (Λ, φ, H_δy, H_δx, ϕ_lw, ϕ_up, off)

Z0 = prd.phase_tilt(*Holo_params)
Z1 = prd.phase_mod(Z0, ϕ_lw, ϕ_up)

holo_data = np.genfromtxt(f0, delimiter=',')
print(holo_data)

y_dB = np.genfromtxt(f1, delimiter=',')
y_lin = np.power(10, y_dB / 10) / np.max(np.power(10, y_dB / 10))

x0 = np.genfromtxt(f2, delimiter=',')
x1 = np.linspace(0, 255, 25)
x3 = np.linspace(0, 255, 256)
fun1 = interp1d(x0, y_lin)
initial_guess = (15, 1 / 800)

try:
    popt, _ = opt.curve_fit(prd.P_g_fun, x1, fun1(
        x1), p0=initial_guess, bounds=([0, -np.inf], [np.inf, np.inf]))

except RuntimeError:
    print("Error - curve_fit failed")
ϕ_A = popt[0]
ϕ_B = popt[1]
ϕ_g_lu = prd.ϕ_g_fun(x3, popt[0], popt[1])
ϕ_max = ϕ_g_lu[-1]
ϕ_g = interp1d(np.linspace(0, 255, 256), ϕ_g_lu)
g_ϕ0 = interp1d(ϕ_g_lu, np.linspace(0, 255, 256))
ϕ1 = np.linspace(0, ϕ_max, 256)
gs0 = g_ϕ0(ϕ1)

g_ind1 = gs0 < g_ϕ0(ϕ_lw + os_lw)
g_ind2 = gs0 > g_ϕ0(ϕ_up - os_up)

gs1 = copy.copy(gs0)
gs2 = copy.copy(gs0)
gs1[g_ind1] = 0
gs2[g_ind2] = 255

gs1 = prd.n_G_blurs(gs1, osw_lw)
gs2 = prd.n_G_blurs(gs2, osw_up)
g_mid = int(g_ϕ0((ϕ_up - ϕ_lw) / 2 + ϕ_lw))

gs3 = np.concatenate((gs1[0:g_mid], gs2[g_mid:]))

g_ϕ1 = interp1d(ϕ1, gs3)

H1 = prd.remap_phase(Z1, g_ϕ1)

Z2 = ϕ_g(H1)

# Increase resolution
# Increase calculation resolution by representing each pixel by px_pad**2
# elements to give a total phase and amplitude field of LC x LC points

LCx = np.shape(Z1)[0] * (2 * px_pad + 1)
LCy = np.shape(Z1)[1] * (2 * px_pad + 1)
LC_field_1 = prd.Pad_A_elements(Z1, px_pad)
LC_field = prd.Pad_A_elements(Z2, px_pad)

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
# Define a point-spread function representing each pixel
R0 = np.zeros((LCx, LCy))
R0[LC_cx - px_pad:LC_cx + px_pad + 1,
    LC_cy - px_pad:LC_cy + px_pad + 1] = 1
R0 = prd.n_G_blurs(R0, 1, px_edge)

# Define new phase profile
phase_SLM_1 = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(LC_field))) * \
    np.fft.fftshift(np.fft.fft2(np.fft.fftshift(R0)))
phase_SLM_2 = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(phase_SLM_1)))


##############################################################################
# aside - plots (i)
##############################################################################

# fig3 = plt.figure('fig3')
# fig3.patch.set_facecolor(cs['mdk_dgrey'])

# ax3_1 = fig3.add_subplot(411)
# plt.plot(H[0, 0: 2 * Λ], 'o')
# plt.plot(np.linspace(0, 2 * Λ, 2 * Λ * (2 * px_pad + 1)) - 0.5,
#          phase_SLM_2[px_pad, 0: 2 * Λ * (2 * px_pad + 1)])

# ax3_3 = fig3.add_subplot(423)
# plt.plot(R0[LC_cy, LC_cx - 2 * px_pad - 1:LC_cx + 2 * px_pad + 2], 'o-')
# plt.plot([px_pad + 1, px_pad + 1], [1, 0], c=cs['ggblue'])
# plt.plot([3 * px_pad + 1, 3 * px_pad + 1], [1, 0], c=cs['ggblue'])

# ax3_4 = fig3.add_subplot(424)
# plt.imshow(R0[LC_cx - 2 * px_pad:LC_cx + 2 * px_pad + 1,
#               LC_cy - 2 * px_pad:LC_cy + 2 * px_pad + 1])
# ax3_4.add_patch(
#     patches.Rectangle((px_pad, px_pad),
#                       2 * px_pad, 2 * px_pad,
#                       fill=False, edgecolor=cs['ggred'], lw=2))

# ax3_5 = fig3.add_subplot(425)
# plt.imshow(LC_field[0:Λ * 2 * px_pad + 1,
#                     0:Λ * 2 * px_pad + 1])

# ax3_6 = fig3.add_subplot(426)
# plt.imshow(abs(phase_SLM_2[0:Λ * 2 * px_pad + 1,
#                            0:Λ * 2 * px_pad + 1]))

# ax3_7 = fig3.add_subplot(414)
# plt.plot(np.linspace(0, H_δx, LCx), E_field[LC_cx, :])

# plt.tight_layout()


##############################################################################
# 4 - Calculate replay field
##############################################################################
# Define phase distribution when there is no hologram displayed
SLM_zero = np.zeros([LCx, LCy])

# Define zero padding factor, pad, and generate associated replay field
# calaculation matrices
E_calc = np.zeros([fft_pad * LCx, fft_pad * LCy])
E_calc_phase = np.zeros([fft_pad * LCx, fft_pad * LCy]) * 0j
E_calc_amplt = E_calc_phase

# Calculation of replay field when no grating is displayed ###################
E_calc_phase[0:LCx, 0:LCy] = SLM_zero[:, :]
E_calc_amplt[0:LCx, 0:LCy] = E_field[:, :]
E_replay_zero = np.fft.fftshift(
    np.fft.fft2(np.fft.fftshift(E_calc_amplt * np.exp(1j * E_calc_phase))))
I_replay_zero = (abs(E_replay_zero))**2

# Maximum intensity
I_max_zero = np.max(np.max(I_replay_zero))

# Normalized replay field
I_replay_zero = I_replay_zero / I_max_zero
E_calc_phase = E_calc * 0j

# Calculation of replay field when grating is displayed ######################
E_calc_phase[0:LCx, 0:LCy] = phase_SLM_2[:, :]
E_calc_amplt[0:LCx, 0:LCy] = E_field[:, :]
E_replay = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(E_calc_amplt * np.exp(
    1j * E_calc_phase))))
I_replay = (abs(E_replay))**2
# Maximum intensity
I_max_signal = np.max(np.max(I_replay))
# Replay intensity distribution normalized with respect to the undiffracted
# zeroth order
I_replay = I_replay / I_max_zero

# Corresponding insertion loss
Loss = I_max_signal / I_max_zero
print('Loss = ', Loss)

I1_final = np.zeros([200, 200])
I1_final = 10 * np.log10(I_replay_zero[int(LCx * fft_pad / 2 - 100):
                                       int(LCx * fft_pad / 2 + 100),
                                       int(LCy * fft_pad / 2 - 100):
                                       int(LCy * fft_pad / 2 + 100)])
I1_final[I1_final < -60] = -60
I1_final_full = 10 * np.log10(I_replay_zero)
I1_final_full[I1_final_full < -60] = -60

I2_final = np.zeros([200, 200])
I2_final = 10 * np.log10(I_replay[int(LCx * fft_pad / 2 - 100):
                                  int(LCx * fft_pad / 2 + 100),
                                  int(LCy * fft_pad / 2 - 100):
                                  int(LCy * fft_pad / 2 + 100)])
I2_final[I2_final < -60] = -60
I2_final_full = 10 * np.log10(I_replay)
I2_final_full[I2_final_full < -60] = -60

# Generate axis
Ratio2 = np.shape(phase_SLM_2)[0] / np.shape(Z1)[0]
Ratio1 = np.shape(I_replay)[0] / np.shape(I2_final)[0]

LCOS_x = H_δx * px
LCOS_y = H_δy * px

RePl_x = (f * λ) / (px / Ratio2)
RePl_y = (f * λ) / (px / Ratio2)


LCOS_x_ax = 1e6 * np.linspace(-LCOS_x, LCOS_x, np.shape(phase_SLM_2)[0])
LCOS_y_ax = 1e6 * np.linspace(-LCOS_y, LCOS_y, np.shape(phase_SLM_2)[1])

FFT_x_ax = (1e6 / Ratio1) * np.linspace(-RePl_x / 2, RePl_x / 2,
                                        np.shape(I2_final)[0])


FFT_y_ax = (1e6 / Ratio1) * np.linspace(-RePl_y / 2, RePl_y / 2,
                                        np.shape(I2_final)[1])

print('I_reply = ', np.shape(I_replay))
print('I2_final = ', np.shape(I2_final))
print('Ratio = ', np.shape(I_replay)[0] / np.shape(I2_final)[0])
print('Ratio2 = ', np.shape(phase_SLM_2)[0] / np.shape(Z1)[0])
print('LCOS size = ', LCOS_x * 1e6)
print('Replay Field size = ', RePl_x * 1e6)
print('Replay Field plot size = ', RePl_x * 1e6 / Ratio1)
f1 = p1 + '\Replay_field.csv'
f2 = p1 + '\Replay_x.csv'
f3 = p1 + '\Replay_y.csv'
np.savetxt(f1, I2_final)
np.savetxt(f2, FFT_x_ax)
np.savetxt(f3, FFT_y_ax)
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

# fig1 = plt.figure('fig1')
# fig1.patch.set_facecolor(cs['mdk_dgrey'])

# ax11 = fig1.add_subplot(221)
# plt.imshow(I1_final_full)

# ax12 = fig1.add_subplot(222)
# plt.imshow(I2_final_full)

# ax13 = fig1.add_subplot(223)
# plt.imshow(np.abs(E_calc_phase))
# plt.title('SLM Phase', fontsize=8)

# ax14 = fig1.add_subplot(224)
# plt.imshow(np.abs(E_calc_amplt))
# plt.title('Incident Beam', fontsize=8)

# plt.tight_layout()

# fig1 = plt.figure('fig1')
# fig1.patch.set_facecolor(cs['mdk_dgrey'])
# ax1 = fig1.add_subplot(111)
# ax1.set_xlabel('x axis')
# ax1.set_ylabel('y axis')

# plt.imshow(I2_final)

fig2 = plt.figure('fig2')
fig2.patch.set_facecolor(cs['mdk_dgrey'])
ax2 = fig2.add_subplot(111)
ax2.set_xlabel('x axis (replay field distance - μm)')
ax2.set_ylabel('y axis (power - dB)')

plt.plot(FFT_x_ax, I2_final[:, np.shape(I2_final)[1] / 2], 'o-')

# fig4 = plt.figure('fig4')
# fig4.patch.set_facecolor(cs['mdk_dgrey'])
# ax4 = fig4.add_subplot(111)
# ax4.set_xlabel('x axis')
# ax4.set_ylabel('y axis')
# plt.plot(H[0, 0: 2 * Λ], 'o')
# plt.plot(np.linspace(0, 2 * Λ, 2 * Λ * (2 * px_pad + 1)) - 0.5,
#          phase_SLM_2[px_pad, 0: 2 * Λ * (2 * px_pad + 1)])

fig5 = plt.figure('fig5')
fig5.patch.set_facecolor(cs['mdk_dgrey'])
ax5_1 = fig5.add_subplot(121)
ax5_1.set_xlabel('LCOS x axis (μm)')
ax5_1.set_ylabel('LCOS y axis (μm)')
plt.imshow(Z2, extent=prd.extents(LCOS_x_ax) + prd.extents(LCOS_y_ax))
ax5_2 = fig5.add_subplot(122)
ax5_2.set_xlabel('Replay x axis (μm)')
ax5_2.set_ylabel('Replay y axis (μm)')
plt.imshow(I2_final, extent=prd.extents(FFT_x_ax) + prd.extents(FFT_y_ax))
plt.tight_layout()

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

plt.show()

# prd.PPT_save_2d(fig1, ax1, 'plot0.png')
# prd.PPT_save_2d(fig2, ax2, 'plot1.png')
# prd.PPT_save_2d(fig4, ax4, 'plot2.png')
