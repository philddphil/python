##############################################################################
# Import some libraries
##############################################################################

import os
import glob
import copy
import random
import io
import csv

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.optimize as opt

from scipy import ndimage
from scipy import io
from scipy.interpolate import interp1d
from scipy.special import erf

from PIL import Image

np.set_printoptions(suppress=True)


###############################################################################
# File & plotting defs
###############################################################################
# Modokai pallette for plotting ###############################################
def palette():
    colours = {'mdk_purple': [145 / 255, 125 / 255, 240 / 255],
               'mdk_dgrey': [39 / 255, 40 / 255, 34 / 255],
               'mdk_lgrey': [96 / 255, 96 / 255, 84 / 255],
               'mdk_green': [95 / 255, 164 / 255, 44 / 255],
               'mdk_yellow': [229 / 255, 220 / 255, 90 / 255],
               'mdk_blue': [75 / 255, 179 / 255, 232 / 255],
               'mdk_orange': [224 / 255, 134 / 255, 31 / 255],
               'mdk_pink': [180 / 255, 38 / 255, 86 / 255],
               ####
               'rmp_dblue': [12 / 255, 35 / 255, 218 / 255],
               'rmp_lblue': [46 / 255, 38 / 255, 86 / 255],
               'rmp_pink': [210 / 255, 76 / 255, 197 / 255],
               'rmp_green': [90 / 255, 166 / 255, 60 / 255],
               ####
               'fibre9l_1': [234 / 255, 170 / 255, 255 / 255],
               'fibre9l_2': [255 / 255, 108 / 255, 134 / 255],
               'fibre9l_3': [255 / 255, 182 / 255, 100 / 255],
               'fibre9l_4': [180 / 255, 151 / 255, 255 / 255],
               'fibre9l_6': [248 / 255, 255 / 255, 136 / 255],
               'fibre9l_7': [136 / 255, 172 / 255, 255 / 255],
               'fibre9l_8': [133 / 255, 255 / 255, 226 / 255],
               'fibre9l_9': [135 / 255, 255 / 255, 132 / 255],
               'fibre9d_1': [95 / 255, 0 / 255, 125 / 255],
               'fibre9d_2': [157 / 255, 0 / 255, 28 / 255],
               'fibre9d_3': [155 / 255, 82 / 255, 0 / 255],
               'fibre9d_4': [40 / 255, 0 / 255, 147 / 255],
               'fibre9d_6': [119 / 255, 125 / 255, 0 / 255],
               'fibre9d_7': [0 / 255, 39 / 255, 139 / 255],
               'fibre9d_8': [0 / 255, 106 / 255, 85 / 255],
               'fibre9d_9': [53 / 255, 119 / 255, 0 / 255],
               ####
               'ggred': [217 / 255, 83 / 255, 25 / 255],
               'ggblue': [30 / 255, 144 / 255, 229 / 255],
               'ggpurple': [145 / 255, 125 / 255, 240 / 255],
               'ggyellow': [229 / 255, 220 / 255, 90 / 255],
               'gglred': [237 / 255, 103 / 255, 55 / 255],
               'gglblue': [20 / 255, 134 / 255, 209 / 255],
               'gglpurple': [165 / 255, 145 / 255, 255 / 255],
               'gglyellow': [249 / 255, 240 / 255, 110 / 255],
               'ggdred': [197 / 255, 63 / 255, 5 / 255],
               'ggdblue': [0 / 255, 94 / 255, 169 / 255],
               'ggdpurple': [125 / 255, 105 / 255, 220 / 255],
               'ggdyellow': [209 / 255, 200 / 255, 70 / 255],

               }

    plt.style.use('ggplot')
    plt.rcParams['font.size'] = 8
    plt.rcParams['font.family'] = 'monospace'
    plt.rcParams['font.fantasy'] = 'Nimbus Mono'
    plt.rcParams['axes.labelsize'] = 8
    plt.rcParams['axes.labelweight'] = 'normal'
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 8
    plt.rcParams['lines.color'] = 'white'
    plt.rcParams['text.color'] = colours['mdk_purple']
    plt.rcParams['axes.labelcolor'] = colours['mdk_yellow']
    plt.rcParams['xtick.color'] = colours['mdk_purple']
    plt.rcParams['ytick.color'] = colours['mdk_purple']
    plt.rcParams['axes.edgecolor'] = colours['mdk_lgrey']
    plt.rcParams['savefig.edgecolor'] = colours['mdk_lgrey']
    plt.rcParams['axes.facecolor'] = colours['mdk_dgrey']
    plt.rcParams['savefig.facecolor'] = colours['mdk_dgrey']
    plt.rcParams['grid.color'] = colours['mdk_lgrey']
    plt.rcParams['grid.linestyle'] = ':'
    plt.rcParams['axes.titlepad'] = 6

    return colours


# Load multiple .csvs #########################################################
def load_multicsv(directory):
    f1 = directory + r'\*.csv'
    files = glob.glob(f1)
    data_all = np.array([])
    for i1, val1 in enumerate(files[0:]):
        data = np.genfromtxt(val1, delimiter=',')
        data_all = np.append(data_all, data)

    return data_all


# Plot an image from a csv ####################################################
def img_csv(file, delim=',', sk_head=1):
    im = np.genfromtxt(file, delimiter=delim, skip_header=sk_head)
    im_size = np.shape(im)
    y = np.arange(im_size[0])
    x = np.arange(im_size[1])
    X, Y = np.meshgrid(x, y)
    coords = (X, Y)
    return (im, coords)


# Plot an image from a labVIEW ################################################
def img_labVIEW(file):
    im = np.loadtxt(file)
    im_size = np.shape(im)
    y = np.arange(im_size[0])
    x = np.arange(im_size[1])
    X, Y = np.meshgrid(x, y)
    coords = (X, Y)
    return (im, coords)


# Save 3d plot as a colourscheme suitable for ppt, as a png ###################
def PPT_save_3d(fig, ax, name):
    plt.rcParams['text.color'] = 'xkcd:charcoal grey'
    fig.patch.set_facecolor('xkcd:white')
    ax.patch.set_facecolor('xkcd:white')
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    ax.xaxis.label.set_color('xkcd:charcoal grey')
    ax.yaxis.label.set_color('xkcd:charcoal grey')
    ax.zaxis.label.set_color('xkcd:charcoal grey')
    ax.tick_params(axis='x', colors='xkcd:charcoal grey')
    ax.tick_params(axis='y', colors='xkcd:charcoal grey')
    ax.tick_params(axis='z', colors='xkcd:charcoal grey')
    fig.savefig(name)


# Save 2d plot as a colourscheme suitable for ppt, as a png ###################
def PPT_save_2d(fig, ax, name):
    plt.rcParams['text.color'] = 'xkcd:charcoal grey'
    plt.rcParams['savefig.facecolor'] = ((1.0, 1.0, 1.0, 0.0))
    ax.patch.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax.xaxis.label.set_color('xkcd:charcoal grey')
    ax.yaxis.label.set_color('xkcd:charcoal grey')
    ax.tick_params(axis='x', colors='xkcd:charcoal grey')
    ax.tick_params(axis='y', colors='xkcd:charcoal grey')

    ax.figure.savefig(name)


# Save 2d image as a colourscheme suitable for ppt, as a png ##################
def PPT_save_2d_im(fig, ax, cb, name):
    plt.rcParams['text.color'] = 'xkcd:charcoal grey'
    plt.rcParams['savefig.facecolor'] = ((1.0, 1.0, 1.0, 0.0))
    ax.patch.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax.xaxis.label.set_color('xkcd:charcoal grey')
    ax.yaxis.label.set_color('xkcd:charcoal grey')
    ax.tick_params(axis='x', colors='xkcd:charcoal grey')
    ax.tick_params(axis='y', colors='xkcd:charcoal grey')
    cbytick_obj = plt.getp(cb.ax.axes, 'yticklabels')
    cbylabel_obj = plt.getp(cb.ax.axes, 'yticklabels')
    plt.setp(cbytick_obj, color='xkcd:charcoal grey')

    ax.figure.savefig(name)


# Smooth a numpy image array ##################################################
def img_clean(im):
    im_size = np.shape(im)
    y = np.arange(im_size[0])
    x = np.arange(im_size[1])
    y1 = np.arange(0, im_size[0], 10)
    x1 = np.arange(0, im_size[1], 10)

    X, Y = np.meshgrid(x, y)
    X1, Y1 = np.meshgrid(x1, y1)

    RBS_f = RectBivariateSpline(y, x, im)
    RBS_im = RBS_f(y1, x1)
    G_RBS_im = gaussian_filter(RBS_im, 10)
    G_RBS = RectBivariateSpline(y1, x1, G_RBS_im)
    smooth_im = G_RBS(y, x)
    return smooth_im


# Save a hologram (nupmy array) to a gray scale bmp ###########################
def save_bmp(X, Path):
    plt.imsave(Path + '.png', X,
               cmap=plt.cm.gray, vmin=0, vmax=255)
    file_in = Path + '.png'
    img = Image.open(file_in)
    file_out = Path + '.bmp'
    img.save(file_out)


# Cross correlating two images, returns the fftconvolution ####################
def cross_image(im1, im2):
    # get rid of the color channels by performing a grayscale transform
    # the type cast into 'float' is to avoid overflows
    im1_gray = im1.astype('float')
    im2_gray = im2.astype('float')

    # get rid of the averages, otherwise the results are not good
    im1_gray -= np.mean(im1_gray)
    im2_gray -= np.mean(im2_gray)

    # calculate the correlation image; note the flipping of onw of the images
    return sp.signal.fftconvolve(im1_gray, im2_gray[::-1, ::-1], mode='same')


#  For us with extents in imshow ##############################################
def extents(f):
    delta = f[1] - f[0]
    return [f[0] - delta / 2, f[-1] + delta / 2]


###############################################################################
# Hologram defs
###############################################################################
# Overshoot mapping ###########################################################
def overshoot_phase(H1, g_OSlw, g_OSup, g_min, g_max):
    H2 = copy.copy(H1)
    Super_thres_indices = H1 > g_OSup
    Sub_thres_indices = H1 <= g_OSlw
    H2[Super_thres_indices] = g_max
    H2[Sub_thres_indices] = g_min
    return (H2)


# Upack values from Hologram control sent by LabVIEW ##########################
def variable_unpack(LabVIEW_data):
    Λ = LabVIEW_data[0]
    φ = LabVIEW_data[1]

    L_δx = LabVIEW_data[2]
    L_δy = LabVIEW_data[3]

    H_δx = LabVIEW_data[4]
    H_δy = LabVIEW_data[5]
    H_cx = LabVIEW_data[6]
    H_cy = LabVIEW_data[7]

    ϕ_min = LabVIEW_data[8]
    ϕ_max = LabVIEW_data[9]
    ϕ_lw = LabVIEW_data[10]
    ϕ_up = LabVIEW_data[11]

    g_min = LabVIEW_data[12]
    g_max = LabVIEW_data[13]
    g_lw = LabVIEW_data[14]
    g_up = LabVIEW_data[15]

    offset = LabVIEW_data[16]
    sin_amp = LabVIEW_data[17]
    sin_off = LabVIEW_data[18]

    params = [Λ, φ, L_δx, L_δy,
              H_δx, H_δy, H_cx, H_cy,
              ϕ_min, ϕ_max, ϕ_lw, ϕ_up,
              g_min, g_max, g_lw, g_up,
              offset, sin_amp, sin_off]
    return params


# Generate hologram and save as bmp ###########################################
def holo_gen(*LabVIEW_data):
    # Unpack parameters
    # cs = palette()
    Λ = LabVIEW_data[0]
    φ = (np.pi / 180) * LabVIEW_data[1]

    L_δx = LabVIEW_data[2]
    L_δy = LabVIEW_data[3]

    H_δx = LabVIEW_data[4]
    H_δy = LabVIEW_data[5]
    H_cx = LabVIEW_data[6]
    H_cy = LabVIEW_data[7]

    ϕ_lw = np.pi * LabVIEW_data[10]
    ϕ_up = np.pi * LabVIEW_data[11]

    g_min = LabVIEW_data[12]
    g_max = LabVIEW_data[13]
    g_lw = LabVIEW_data[14]
    g_up = LabVIEW_data[15]

    offset = LabVIEW_data[16]

    sin_amp = LabVIEW_data[17]
    sin_off = LabVIEW_data[18]

    # Phase mapping details (ϕ)
    ϕ_g = fit_phase()
    g_ϕ = interp1d(ϕ_g, np.linspace(0, 255, 256))
    ϕ_max = ϕ_g[-1]
    # Define holo params
    L_δyx = (L_δy, L_δx)
    H_δyx = (H_δy, H_δx)
    H_cyx = (H_cy, H_cx)
    ϕ_lims = (ϕ_lw, ϕ_up)
    Holo_params = (Λ, φ, *H_δyx, *ϕ_lims, offset)

    # Calculate sub hologram (Holo_s)
    Z1 = phase_tilt(*Holo_params)
    Z2 = phase_sin(*Holo_params, sin_amp, sin_off)
    Z_mod = phase_mod(Z2 + Z1, *ϕ_lims)

    # Remap phase with non linear ϕ map

    ϕ1 = np.linspace(0, ϕ_max, 256)
    gs0 = g_ϕ(ϕ1)

    g_ind1 = gs0 < g_ϕ(ϕ_lw + 0.1)
    g_ind2 = gs0 > g_ϕ(ϕ_up - 0.5)

    gs0[g_ind1] = g_min
    gs0[g_ind2] = g_max

    gs0 = n_G_blurs(gs0, 0.5)

    g_ϕ1 = interp1d(ϕ1, gs0)
    H1 = remap_phase(Z_mod, g_ϕ1)
    # Calculate full holograms (Holo_f)
    H2 = add_holo_LCOS(*H_cyx, H1, *L_δyx)

    # Save output
    save_bmp(H2, r"..\..\Data\bmps\hologram")

    # Get phase profile plots and save (use angle of ϕ = π/2 for plotting)
    Z1_0 = phase_tilt(Λ, np.pi / 2, *H_δyx, *ϕ_lims, offset)
    Z2_0 = phase_sin(Λ, np.pi / 2, *H_δyx, *ϕ_lims, offset, sin_amp, sin_off)
    Z2_0_mod = phase_mod(Z2_0 + Z1_0, *ϕ_lims)
    Z1_0_mod = phase_mod(Z1_0, *ϕ_lims)
    h1_0 = remap_phase(Z1_0_mod, g_ϕ1)[:, 0]
    h2_0 = remap_phase(Z2_0_mod, g_ϕ1)[:, 0]
    h3_0 = remap_phase(Z2_0_mod, g_ϕ)[:, 0]

    np.savetxt(r'..\..\Data\Calibration files\greyprofile1.csv',
               h1_0, delimiter=',')
    np.savetxt(r'..\..\Data\Calibration files\greyprofile2.csv',
               h2_0, delimiter=',')
    np.savetxt(r'..\..\Data\Calibration files\greyprofile3.csv',
               h3_0, delimiter=',')

    return [H1]


# Generate holograms with first two parameters to optimise - Λ and φ ##########
def phase_tilt(Λ, φ, H_δy=50, H_δx=50, ϕ_lwlim=0, ϕ_uplim=2 * np.pi, off=0):
    # Generate meshgrid of coordinate points
    x = np.arange(H_δx)
    y = np.arange(H_δy)
    [X, Y] = np.meshgrid(x, y)

    # Calculate phase tilt angle from periodicity and usable phase range
    θ = np.arctan((ϕ_uplim - ϕ_lwlim) / Λ)

    # Convert offset from pixels into phase
    of1 = off * (ϕ_uplim - ϕ_lwlim) / Λ

    # Calculate tilted (unmodulated) phase profile
    Z = np.tan(θ) * (X * np.cos(φ) + Y * np.sin(φ)) - of1

    # Output all 4
    return Z


# Generate phase front with sinusiodal term (related to Λ) rotated by φ #######
def phase_sin(Λ, φ, H_δy, H_δx, ϕ_lwlim, ϕ_uplim, off, sin_amp, sin_off):
    # Generate meshgrid of coordinate points
    x = np.arange(H_δx)
    y = np.arange(H_δy)
    [X, Y] = np.meshgrid(x, y)

    # Calulate higher frequency sinsusoidal profile
    Z = sin_amp * np.sin((4 * np.pi / Λ) * (
                         (X * np.cos(φ) + Y * np.sin(φ)) -
                         sin_off -
                         off))
    return Z


# Modulate a phase front ######################################################
def phase_mod(Z, ϕ_lwlim=0, ϕ_uplim=2 * np.pi):
    δϕ = ϕ_uplim - ϕ_lwlim - 0.00000001
    Z_mod = Z % (δϕ)
    Z_mod = Z_mod + ϕ_lwlim

    return Z_mod


# Calculate replay field for hologram H #######################################
def holo_replay(H, px_edge=1, w=30, px_pad=8, fft_pad=4):
    # Add sub hologram Z_mod to larger hologram (initially set to 0s) ########

    LCx = np.shape(H)[0] * (2 * px_pad + 1)
    LCy = np.shape(H)[1] * (2 * px_pad + 1)
    LC_field = Pad_A_elements(H, px_pad)
    SLM_x = range(LCx)
    SLM_y = range(LCy)
    coords = np.meshgrid(SLM_x, SLM_y)

    # 2 - Define Input field (Here it's Gaussian) intensity profile (2D)
    G1 = Gaussian_2D(coords, 1, LCx / 2, LCy / 2,
                     0.25 * w * (2 * px_pad + 1),
                     0.25 * w * (2 * px_pad + 1))
    E_field = np.reshape(G1, (LCx, LCy))
    (LC_cx, LC_cy) = max_i_2d(E_field)

    # 3 - Generate PSF for pixel diffraction
    R0 = np.zeros((LCx, LCy))
    R0[LC_cx - px_pad:LC_cx + px_pad + 1,
        LC_cy - px_pad:LC_cy + px_pad + 1] = 1
    R0 = n_G_blurs(R0, 1, px_edge)

    # Define new phase profile
    phase_SLM_1 = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(LC_field))) * \
        np.fft.fftshift(np.fft.fft2(np.fft.fftshift(R0)))
    phase_SLM_2 = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(phase_SLM_1)))

    # 4 - Calculate replay field
    # Define phase distribution when there is no hologram displayed
    SLM_zero = np.zeros([LCx, LCy])

    # Define zero padding factor, pad, and generate associated replay field
    # calaculation matrices
    E_calc = np.zeros([fft_pad * LCx, fft_pad * LCy])
    E_calc_phase = np.zeros([fft_pad * LCx, fft_pad * LCy]) * 0j
    E_calc_amplt = E_calc_phase

    # Calculation of replay field when no grating is displayed ###############
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

    # Calculation of replay field when grating is displayed ##################
    E_calc_phase[0:LCx, 0:LCy] = phase_SLM_2[:, :]
    E_calc_amplt[0:LCx, 0:LCy] = E_field[:, :]
    E_replay = np.fft.fftshift(
        np.fft.fft2(np.fft.fftshift(E_calc_amplt * np.exp(1j * E_calc_phase))))
    I_replay = (abs(E_replay))**2
    # Maximum intensity
    # Replay intensity distribution normalized with respect to the undiffracted
    # zeroth order
    I_replay = I_replay / I_max_zero

    I1_final = np.zeros([200, 200])
    I1_final = 10 * np.log10(I_replay_zero[int(LCx * fft_pad / 2 - 100):
                                           int(LCx * fft_pad / 2 + 100),
                                           int(LCy * fft_pad / 2 - 100):
                                           int(LCy * fft_pad / 2 + 100)])
    I1_final[I1_final < -60] = -60

    I2_final = np.zeros([200, 200])
    I2_final = 10 * np.log10(I_replay[int(LCx * fft_pad / 2 - 100):
                                      int(LCx * fft_pad / 2 + 100),
                                      int(LCy * fft_pad / 2 - 100):
                                      int(LCy * fft_pad / 2 + 100)])
    I2_final[I2_final < -60] = -60

    return I2_final


# Add sub-hologram (size [Holδx, Holδy]) to LCOS, size LCOSy, LCOSx ###########
def add_holo_LCOS(H_cy, H_cx, Z_mod, LCOSy, LCOSx):
    LCOSy = int(LCOSy)
    LCOSx = int(LCOSx)
    b0 = np.array([0, 255])
    Holo_f = np.tile(b0, (LCOSy, int(LCOSx / len(b0))))
    # Holo_f = np.zeros((LCOSy,LCOSx))
    dy1 = 0
    dy2 = 0
    dx1 = 0
    dx2 = 0

    (H_δy, H_δx) = np.shape(Z_mod)
    y1 = np.int(H_cy - np.floor(H_δy / 2))
    if y1 < 0:
        dy1 = -1 * y1
        y1 = 0

    y2 = np.int(H_cy + np.ceil(H_δy / 2))
    if y2 > LCOSy:
        dy2 = y2 - LCOSy
        y2 = LCOSy

    x1 = np.int(H_cx - np.floor(H_δx / 2))
    if x1 < 0:
        dx1 = -1 * x1
        x1 = 0
    x2 = np.int(H_cx + np.ceil(H_δx / 2))
    if x2 > LCOSx:
        dx2 = x2 - LCOSx
        x2 = LCOSx

    Holo_f[y1:y2, x1:x2] = Z_mod[dy1:H_δy - dy2, dx1:H_δx - dx2]

    return Holo_f


# Defining the function of Power diffracted as a function of greylevel / P(g) #
def P_g_fun(g, A, B):
    P = np.square(np.sin(A * (1 - np.exp(-B * g))))
    return P


# Defining the function of Power diffracted as a function of ϕ / P(ϕ) #########
def P_ϕ_fun(ϕ):
    P = np.square(np.sin(ϕ))
    return P


# Define the function of phase to greyscale / ϕ(g) ############################
def ϕ_g_fun(g, A, B):
    ϕ = 2 * A * (1 - np.exp(-B * g))
    return ϕ


# Define the function of greyscale to phase / g(ϕ) ############################
def g_ϕ_fun(ϕ, A, B):
    g = np.log(1 - ϕ / (2 * A)) / B
    return g


# Use g(ϕ) defined in 'phase' to fit experimentally obtained phaseramps #######
def fit_phase():
    # f1 = r'C:\Users\Philip\Documents\LabVIEW\Data\Calibration
    # files\Phaseramp.mat'
    f1 = r'..\..\Data\Calibration files\Phase Ps.csv'
    f2 = r'..\..\Data\Calibration files\Phase greys.csv'

    y_dB = np.genfromtxt(f1, delimiter=',')
    y_lin = np.power(10, y_dB / 10) / np.max(np.power(10, y_dB / 10))

    x0 = np.genfromtxt(f2, delimiter=',')
    x1 = np.linspace(0, 255, 25)
    x3 = np.linspace(0, 255, 256)
    f1 = interp1d(x0, y_lin)
    initial_guess = (15, 1 / 800)

    try:
        popt, _ = opt.curve_fit(P_g_fun, x1, f1(
            x1), p0=initial_guess, bounds=([0, -np.inf], [np.inf, np.inf]))

    except RuntimeError:
        print("Error - curve_fit failed")

    ϕ_g = ϕ_g_fun(x3, popt[0], popt[1])

    return ϕ_g


# Use the fitting results from 'fit_phase' to remap hologram Z_mod ############
def remap_phase(Z_mod, g_ϕ):
    H = copy.copy(Z_mod)
    for i1 in range(np.shape(Z_mod)[0]):
        H[i1, :] = g_ϕ(Z_mod[i1, :])
    return (H)


# Use binary search algorithm to find beam on the LCOS ########################
def locate_beam(values, last_CT400, current_CT400, axis):
    # Specify paths of 2 iterators and 1 binary search map
    i0_p = r'..\..\Data\Python loops\Find beam i0.txt'
    i1_p = r'..\..\Data\Python loops\Find beam i1.txt'
    Map_p = r'..\..\Data\Python loops\LCOS Map.txt'

    # Load relevant values
    i0 = np.genfromtxt(i0_p, dtype='int')
    i1 = np.genfromtxt(i1_p, dtype='int')
    Map = np.genfromtxt(Map_p, dtype='int', delimiter=',')

    # Ensure Map is at least a 1d array
    Map = np.atleast_1d(Map)

    # Calculate fractional width of hologram (hd)
    hd = 1 / 2**(i1)

    # Specify axis (x/y)
    if axis == 0:
        LCOS_d_val = 3
        Hol_c_val = 7
        Hol_d_val = 5

    elif axis == 1:
        LCOS_d_val = 2
        Hol_c_val = 6
        Hol_d_val = 4

    # Start search
    start = 0.5
    shift = 0

    # Works on 3 cases i0 == [0,1,2]
    # i0 determines first level of iteration - i.e. 1st or 2nd half of region
    # being checked. 0 case is an exception as it's the first reading
    if i0 == 0:
        values[Hol_c_val] = np.round((start - hd / 2) * values[LCOS_d_val])
        values[Hol_d_val] = np.floor((hd) * values[LCOS_d_val])
        f0 = open(i0_p, 'w')
        f0.write(str(i0 + 1))
        f0.close()

    elif i0 == 1:
        # for i0 == 1 the 2nd measurement of the region is performed
        # i1 determines the region being considered -
        # i1 = 1 is whole LCOS
        # i1 = 2 is 1/2 of LCOS
        # i1 = 3 is 1/4 of LCOS...
        if i1 == 1:
            start = 0.5
        else:
            for j1 in range(i1 - 1):
                shift = -(1 / (2**(j1 + 2))) * (-1)**(Map[j1]) + shift
                start = 0.5 + shift

        values[Hol_c_val] = np.round((start + hd / 2) * values[LCOS_d_val])
        values[Hol_d_val] = np.floor((hd) * values[LCOS_d_val])
        f0 = open(i0_p, 'w')
        f0.write(str(i0 + 1))
        f0.close()

    # for i0 == 2 the power readings for the current and last power values
    # are compared. Depending on outcome the MAP file is duely appended
    elif i0 == 2:

        # Compare the current power with the last power and save outcome
        # to 'Map'
        if current_CT400 < last_CT400:
            Map = np.atleast_1d(np.append(Map, 0))

        elif current_CT400 > last_CT400:
            Map = np.atleast_1d(np.append(Map, 1))

        np.savetxt(Map_p, Map, fmt='%d', delimiter=',')
        i1 = i1 + 1
        hd = 1 / 2**(i1)

        # Prepare the next hologram to be shown
        for j1 in range(i1 - 1):
            shift = -(1 / (2**(j1 + 2))) * (-1)**(Map[j1]) + shift
            start = 0.5 + shift

        values[Hol_c_val] = np.round((start - hd / 2) * values[LCOS_d_val])
        values[Hol_d_val] = np.floor((hd) * values[LCOS_d_val])

        f0 = open(i0_p, 'w')
        f0.write(str(i0 - 1))
        f0.close()

        f1 = open(i1_p, 'w')
        f1.write(str(i1))
        f1.close()

        # Termination statement. Search proceeds whilst i1 <= 8
    if i1 > 8:
        loop_out = 1
    else:
        loop_out = 0
    return(loop_out)


# Use simulated annealing to optimise hologram ###############################
def anneal_H1(values, Ps_last, Ps_current):
    i0_p = r'..\..\Data\Python loops\Anneal i0.txt'
    MF_p = r'..\..\Data\Python loops\Anneal MF.txt'
    XT_p = r'..\..\Data\Python loops\Anneal XT.txt'
    IL_p = r'..\..\Data\Python loops\Anneal IL.txt'
    H_an_p = r'..\..\Data\Python loops\Anneal Hol.txt'
    H_an_pL = r'..\..\Data\Python loops\Anneal Hol Last.txt'

    MF_last = merit(Ps_last)
    MF_current = merit(Ps_current)
    print('Ps last = ', Ps_last)
    print('Ps current = ', Ps_current)
    i0 = np.genfromtxt(i0_p, dtype='int')
    print(i0)
    if i0 == 0:
        H_an = np.random.randint(255, size=(100, 100))
        np.savetxt(H_an_pL, H_an)
        print('Save 1st Holo')
    elif i0 == 1:
        H_an = np.random.randint(255, size=(100, 100))
        np.savetxt(H_an_p, H_an)
        print('Save 2nd Holo')
        data_str = str(MF_current)
        f1 = open(MF_p, 'a')
        f1.write(data_str)
        f1.close()
    else:
        random_x = np.random.choice(int(values[2]))
        random_y = np.random.choice(int(values[3]))
        # Compare last and current MFs
        if MF_current > MF_last:
            H_an = np.genfromtxt(H_an_p, dtype='int')
            np.savetxt(H_an_pL, H_an)
            H_an[random_x, random_y] = np.random.randint(0, 255)
            np.savetxt(H_an_p, H_an)
        else:
            H_an = np.genfromtxt(H_an_pL, dtype='int')
            H_an[random_x, random_y] = np.random.randint(0, 255)
            np.savetxt(H_an_p, H_an)
            MF_str = ',' + str(MF_current)
            f1 = open(MF_p, 'a')
            f1.write(MF_str)
            f1.close()
            XT_str = ',' + str(Ps_current[0] - Ps_current[1])
            f2 = open(XT_p, 'a')
            f2.write(XT_str)
            f2.close()
            IL_str = ',' + str(Ps_current[0])
            f3 = open(IL_p, 'a')
            f3.write(IL_str)
            f3.close()

    Holo_f = add_holo_LCOS(values[5], values[4], H_an,
                           values[1], values[0])
    save_bmp(Holo_f, r"..\..\Data\bmps\hologram")
    i0 = i0 + 1
    f0 = open(i0_p, 'w')
    f0.write(str(i0))
    f0.close()

    # Termination statement. Search proceeds whilst i1 <= 8
    if MF_current > -6:
        loop_out = 1
    else:
        loop_out = 0
    return(loop_out)


# Use simulated annealing to optimise hologram ###############################
def anneal_H2(values, Ps_last, Ps_current, H_in):
    i0_p = r'..\..\Data\Python loops\Anneal i0.txt'
    MF_p = r'..\..\Data\Python loops\Anneal MF.txt'
    XT_p = r'..\..\Data\Python loops\Anneal XT.txt'
    IL_p = r'..\..\Data\Python loops\Anneal IL.txt'
    H_an_p = r'..\..\Data\Python loops\Anneal Hol.txt'
    H_an_pL = r'..\..\Data\Python loops\Anneal Hol Last.txt'
    MF_last = merit(Ps_last)
    MF_current = merit(Ps_current)
    i0 = np.genfromtxt(i0_p, dtype='int')
    print(i0)
    if i0 == 0:
        H_an = H_in
        np.savetxt(H_an_pL, H_an)
        print('Save 1st Holo')
    elif i0 == 1:
        H_an = H_in
        np.savetxt(H_an_p, H_an)
        print('Save 2nd Holo')
        data_str = str(MF_current)
        f1 = open(MF_p, 'a')
        f1.write(data_str)
        f1.close()
    else:
        random_x = np.random.choice(int(values[2]))
        random_y = np.random.choice(int(values[3]))
        # Compare last and current MFs
        if MF_current > MF_last:
            H_an = np.genfromtxt(H_an_p, dtype='int')
            np.savetxt(H_an_pL, H_an)
            H_an[random_x, random_y] = np.random.randint(0, 255)
            np.savetxt(H_an_p, H_an)
            print('∆MF ++++++++ Change')
        else:
            H_an = np.genfromtxt(H_an_pL, dtype='int')
            H_an[random_x, random_y] = np.random.randint(0, 255)
            np.savetxt(H_an_p, H_an)
            MF_str = ',' + str(MF_current)
            f1 = open(MF_p, 'a')
            f1.write(MF_str)
            f1.close()
            XT_str = ',' + str(Ps_current[0] - Ps_current[1])
            f2 = open(XT_p, 'a')
            f2.write(XT_str)
            f2.close()
            IL_str = ',' + str(Ps_current[0])
            f3 = open(IL_p, 'a')
            f3.write(IL_str)
            f3.close()
            print('∆MF ------- Dont Change')

    Holo_f = add_holo_LCOS(values[5], values[4], H_an,
                           values[1], values[0])
    save_bmp(Holo_f, r"..\..\Data\bmps\hologram")
    i0 = i0 + 1
    f0 = open(i0_p, 'w')
    f0.write(str(i0))
    f0.close()

    # Termination statement. Search proceeds whilst i1 <= 8
    if MF_current > -6:
        loop_out = 1
    else:
        loop_out = 0
    return(loop_out)


# Use simulated annealing to optimise hologram ###############################
def anneal_H3(values, Ps_current, variables):
    i0_p = r'..\..\Data\Python loops\Anneal i0.txt'
    MF_p = r'..\..\Data\Python loops\Anneal MF.txt'
    MFk_p = r'..\..\Data\Python loops\Anneal MF keep.txt'
    XT_p = r'..\..\Data\Python loops\Anneal XT.txt'
    IL_p = r'..\..\Data\Python loops\Anneal IL.txt'
    H_an_p = r'..\..\Data\Python loops\Anneal Hol params.txt'
    H_an_pk = r'..\..\Data\Python loops\Anneal Hol params keep.txt'
    MFk = np.genfromtxt(MFk_p)
    MF_current = merit(Ps_current)
    i0 = np.genfromtxt(i0_p, dtype='int')

    print(i0)

    ϕ_lwlim_rng = (values[6], min(values[7], values[8] + 0.1))
    ϕ_uplim_rng = (values[9] - 0.1, min(values[7], values[9] + 0.1))
    g_OSlw_rng = (values[12], values[12] + 1)
    g_OSup_rng = (values[13] - 1, values[13])
    g_min_rng = (0, values[10])
    g_max_rng = (values[11], 255)
    Λ_rng = (values[14] - 0.1, values[14] + 0.1)
    φ_rng = (values[15] - 0.2, values[15] + 0.2)
    offset_rng = (0, values[14])
    sin_amp_rng = (0, 0.2)
    sin_off_rng = (0, values[14])

    params_to_vary = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    rngs_to_vary = [ϕ_lwlim_rng, ϕ_uplim_rng, g_OSlw_rng, g_OSup_rng,
                    g_min_rng, g_max_rng, Λ_rng, φ_rng, offset_rng,
                    sin_amp_rng, sin_off_rng]

    if i0 == 0:
        np.savetxt(H_an_pk, values, delimiter=",",
                   header='see code structure for variable names')
        random_param = random.choice(range(len(params_to_vary)))
        param_to_vary = params_to_vary[random_param]
        rng_to_vary = rngs_to_vary[random_param]
        new_value = np.random.uniform(rng_to_vary[0], rng_to_vary[1])
        values[param_to_vary] = new_value
        print(values)
        print('Param changed is', variables[param_to_vary])
        print('New value is', new_value)
        np.savetxt(H_an_p, values, delimiter=",",
                   header='see code structure for variable names')
        f0 = open(MFk_p, 'w')
        f0.write(str(MF_current))
        f0.close()
        holo_gen(*values)
    else:
        vs_keep = np.genfromtxt(H_an_pk, delimiter=',')
        vs_current = np.genfromtxt(H_an_p, delimiter=',')

        random_param = random.choice(range(len(params_to_vary)))
        param_to_vary = params_to_vary[random_param]
        rng_to_vary = rngs_to_vary[random_param]
        new_value = np.random.uniform(rng_to_vary[0], rng_to_vary[1])
        print('Kept-', MFk, ' / ', 'Current-', MF_current)
        # Compare last and current MFs

        if MF_current > MFk:
            vs_current[param_to_vary] = new_value
            print('Param changed is', variables[param_to_vary])
            print('New value is', new_value)
            np.savetxt(H_an_pk, vs_current, delimiter=",",
                       header='see code structure for variable names')
            new_values = vs_current
            f0 = open(MFk_p, 'w')
            f0.write(str(MF_current))
            f0.close()
            MF_str = ',' + str(MF_current)
            f1 = open(MF_p, 'a')
            f1.write(MF_str)
            f1.close()
            XT_str = ',' + str(Ps_current[0] - Ps_current[1])
            f2 = open(XT_p, 'a')
            f2.write(XT_str)
            f2.close()
            IL_str = ',' + str(Ps_current[0])
            f3 = open(IL_p, 'a')
            f3.write(IL_str)
            f3.close()
            print('∆MF ++++++++ Change (Current > Kept)')
        else:
            vs_keep[param_to_vary] = new_value
            new_values = vs_keep

        np.savetxt(H_an_p, new_values, delimiter=",",
                   header='see code structure for variable names')

        holo_gen(*new_values)
    i0 = i0 + 1
    f0 = open(i0_p, 'w')
    f0.write(str(i0))
    f0.close()
    # Termination statement. Search proceeds whilst i1 <= 8
    if MF_current > -6:
        loop_out = 1
    else:
        loop_out = 0
    return(loop_out)


# Sweep parameters and select optimal value ##################################
def sweep(values, Ps_current, variables, param=0):
    i0_p = r'..\..\Data\Python loops\Swept i0.txt'
    MF_p = r'..\..\Data\Python loops\Swept MF.txt'
    XT_p = r'..\..\Data\Python loops\Swept XT.txt'
    IL_p = r'..\..\Data\Python loops\Swept IL.txt'
    Rng_p = r'..\..\Data\Python loops\Swept Rng.txt'
    H_swp_p = r'..\..\Data\Python loops\Sweep H.txt'
    param_swp_p = r'..\..\Data\Python loops\Swept param.txt'

    pts = 20

    MF_current = merit(Ps_current)
    i0 = np.genfromtxt(i0_p, dtype='int')
    rng = np.genfromtxt(param_swp_p)
    param_2_swp = int(param)

    print('sweep pt - ', i0)

    if i0 == 0:
        Λ_rng = (values[0] - 0.5, values[0] + 0.5)
        φ_rng = (values[1] - 1, values[1] + 1)
        H_δx_rng = []
        H_δy_rng = []
        H_cx_rng = []
        H_cy_rng = []

        ϕ_lw_rng = (max(values[8], 0.9 * values[10]),
                    min(values[9], 1.1 * values[10]))
        ϕ_up_rng = (max(values[8], 0.9 * values[11]),
                    min(values[9], 1.1 * values[11]))

        g_min_rng = (max(0, values[12] - 10),
                     min(255, values[12] + 10))
        g_max_rng = (max(0, values[13] - 10),
                     min(255, values[13] + 10))

        g_lw_rng = (max(0, values[14] - 10),
                    min(255, values[14] + 10))
        g_up_rng = (max(0, values[15] - 10),
                    min(255, values[15] + 10))

        offset_rng = (0, values[14] / 5)
        sin_amp_rng = (0, 0.2)
        sin_off_rng = (0, values[14] / 5)
        all_rngs = [Λ_rng,
                    φ_rng,
                    0,
                    0,
                    H_δx_rng,
                    H_δy_rng,
                    H_cx_rng,
                    H_cy_rng,
                    0,
                    0,
                    ϕ_lw_rng,
                    ϕ_up_rng,
                    g_min_rng,
                    g_max_rng,
                    g_lw_rng,
                    g_up_rng,
                    offset_rng,
                    sin_amp_rng,
                    sin_off_rng]
        rng_2_swp = all_rngs[param_2_swp]
        rng = np.linspace(rng_2_swp[0], rng_2_swp[1], pts)
        np.savetxt(param_swp_p, rng, delimiter=',')
        new_value = rng[i0]
        values[param_2_swp] = new_value
        np.savetxt(H_swp_p, values, delimiter=",",
                   header='see code structure for variable names')
        holo_gen(*values)

    elif i0 == pts:
        MF_str = str(MF_current)
        f1 = open(MF_p, 'a')
        f1.write(MF_str)
        f1.close()
        XT_str = str(Ps_current[0] - Ps_current[1])
        f2 = open(XT_p, 'a')
        f2.write(XT_str)
        f2.close()
        IL_str = str(Ps_current[0])
        f3 = open(IL_p, 'a')
        f3.write(IL_str)
        f3.close()
        Rng_str = str(rng[i0 - 1])
        f4 = open(Rng_p, 'a')
        f4.write(Rng_str)
        f4.close()
    else:
        new_value = rng[i0]
        values[param_2_swp] = new_value
        print('New value is', new_value)
        MF_str = str(MF_current) + ','
        f1 = open(MF_p, 'a')
        f1.write(MF_str)
        f1.close()
        XT_str = str(Ps_current[0] - Ps_current[1]) + ','
        f2 = open(XT_p, 'a')
        f2.write(XT_str)
        f2.close()
        IL_str = str(Ps_current[0]) + ','
        f3 = open(IL_p, 'a')
        f3.write(IL_str)
        f3.close()
        Rng_str = str(rng[i0 - 1]) + ','
        f4 = open(Rng_p, 'a')
        f4.write(Rng_str)
        f4.close()
        np.savetxt(H_swp_p, values, delimiter=",",
                   header='see code structure for variable names')
        holo_gen(*values)

    # Termination statement. Search proceeds whilst i1 <= 8
    if i0 == pts:
        loop_out = 1
        print('End sweep')
    else:
        loop_out = 0
        print('Carry on')

    i0 = i0 + 1
    f0 = open(i0_p, 'w')
    f0.write(str(i0))
    f0.close()
    return loop_out, values


# Basic merit function (to be developed) #####################################
def merit(Ps):

    IL = Ps[0]
    XT = Ps[0] - Ps[1]
    MF = IL - (55 - XT)
    return MF


# Fit the sweep results ######################################################
def sweep_fit():
    p1 = (r"..\..\Data\Python loops")
    f3 = r"\Swept MF.txt"
    f5 = r"\Swept param.txt"

    MF = np.genfromtxt(p1 + f3, delimiter=',')
    v = np.genfromtxt(p1 + f5, delimiter=',')

    initial_guess = (10, np.mean(v), 0.05, -45)
    try:
        popt, _ = opt.curve_fit(Gaussian_1D, v, MF,
                                p0=initial_guess,
                                bounds=([-np.inf, -np.inf, -np.inf, -np.inf],
                                        [np.inf, np.inf, np.inf, np.inf]))

    except RuntimeError:
        print("Error - curve_fit failed")
        popt = [0, np.mean(v), 0]
    return popt[1]


###############################################################################
# Maths defs
###############################################################################
# Generic 1D Gaussian function ################################################
def Gaussian_1D(x, A, x_c, σ_x, bkg=0, N=1):
    x_c = float(x_c)
    g = bkg + A * np.exp(- (((x - x_c) ** 2) / (2 * σ_x ** 2))**N)
    return g


# Generic 2D Gaussian function ################################################
def Gaussian_2D(coords, A, x_c, y_c, σ_x, σ_y, θ=0, bkg=0, N=1):
    x, y = coords
    x_c = float(x_c)
    y_c = float(y_c)
    a = (np.cos(θ) ** 2) / (2 * σ_x ** 2) + (np.sin(θ) ** 2) / (2 * σ_y ** 2)
    b = -(np.sin(2 * θ)) / (4 * σ_x ** 2) + (np.sin(2 * θ)) / (4 * σ_y ** 2)
    c = (np.sin(θ) ** 2) / (2 * σ_x ** 2) + (np.cos(θ) ** 2) / (2 * σ_y ** 2)
    g = (bkg + A * np.exp(- (a * ((x - x_c) ** 2) +
                             2 * b * (x - x_c) * (y - y_c) +
                             c * ((y - y_c) ** 2))**N))
    return g.ravel()


# Fit Λ and ϕ datasets from peak finding routine ##############################
def find_fit_peak(x, y, A, x_c):
    x_1 = np.linspace(min(x), max(x), 100)
    Peak_ind = np.unravel_index(y.argmax(), y.shape)
    initial_guess = (A, x[Peak_ind[0]], x_c, 0)

    # Fit data
    try:
        popt, pcov = opt.curve_fit(
            Gaussian_1D, x, y, p0=initial_guess)
        fit0 = Gaussian_1D(x_1, *popt)
        p1 = r'C:\Users\User\Documents\Phils LabVIEW\Data\Calibration files\sweepfit.csv'
        p2 = r'C:\Users\User\Documents\Phils LabVIEW\Data\Calibration files\sweepdata.csv'
        np.savetxt(p1, np.column_stack((x_1, fit0)), delimiter=',')
        np.savetxt(p2, np.column_stack((x, y)), delimiter=',')
        Peak_ind_f = np.unravel_index(fit0.argmax(), fit0.shape)
        x_peak = x_1[Peak_ind_f[0]]
        # plt.plot(x_peak, np.max(fit0), 'x', c='xkcd:blue')
        # plt.plot(x_1, fit0, '-', c='xkcd:light blue')
        # plt.draw()
    except RuntimeError:
        print("Error - curve_fit failed")
        x_peak = 0
    return (x_peak)


# Calculate the running mean of N adjacent elements of the array x ############
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N


# Gaussian blur an image n times ##############################################
def n_G_blurs(im, s=3, n=1):
    im_out = im
    for i1 in range(n):
        im_out = ndimage.filters.gaussian_filter(im_out, s)

    return im_out


# Find the RMS of array a #####################################################
def rms(a):
    b = (np.sqrt(np.mean(np.square(a))))
    return b


# Polar to cartesian coords ###################################################
def pol2cart(ρ, ϕ):
    x = ρ * np.cos(ϕ)
    y = ρ * np.sin(ϕ)
    return(x, y)


# Cartesian to polar coords ###################################################
def cart2pol(x, y):
    ρ = np.sqrt(x**2 + y**2)
    ϕ = np.arctan2(y, x)
    return(ρ, ϕ)


# Return the element location of the max of array a ###########################
def max_i_2d(a):
    b = np.unravel_index(a.argmax(), a.shape)
    return(b)


# Circle at location x, y radius r ############################################
def circle(r, x, y):
    # theta goes from 0 to 2pi
    theta = np.linspace(0, 2 * np.pi, 100)

    # compute xc and yc
    xc = r * np.cos(theta) + x
    yc = r * np.sin(theta) + y
    return (xc, yc)


# Mode overlap for 2 fields G1 G2 in field with x & y axis
def Overlap(x, y, G1, G2):
    η1 = sp.trapz(sp.trapz((G1 * G2), y), x)**2
    η2 = sp.trapz(sp.trapz(G1**2, y), x) * sp.trapz(sp.trapz(G2**2, y), x)
    η = η1 / η2
    return η


# Pad an array A with n elements all of value a
def Pad_A_elements(A, n, a=0):
    Ax, Ay = np.shape(A)
    P = a * np.ones(((2 * n + 1) * (Ax), (2 * n + 1) * (Ay)))
    Px, Py = np.shape(P)
    for i1 in range(Px):
        for i2 in range(Py):
            if ((i1 - n) % (2 * n + 1) == 0 and
                    (i2 - n) % (2 * n + 1) == 0):
                P[i1, i2] = A[(i1 - n) // (2 * n + 1),
                              (i2 - n) // (2 * n + 1)]
    return P


###############################################################################
# ABCD matrix defs
###############################################################################
def ABCD_d(q_in, d, n=1):
    M = np.array([[1, d * n], [0, 1]])
    q_out = np.matmul(M, q_in)
    return(q_out)


def ABCD_propagate(q0, z_end, z_start=0, res=100, n=1):
    qz = [q0]
    zs = np.linspace(z_start, z_end, res)
    ns = n * np.ones(len(zs))
    if q0[1] == 1:
        z_start = np.real(q0[0])

    dz = (z_end - z_start) / res

    for i1, val1 in enumerate(zs[1:]):
        q1 = ABCD_d(q0, dz)
        qz.append(q1)
        q0 = q1

    return(zs, qz, ns)


def ABCD_tlens(q_in, f):
    M = np.array([[1, 0], [-1 / f, 1]])
    q_out = np.matmul(M, q_in)
    if q_in[1] == 1:
        q_out = q_out / q_out[1]
    return(q_out)


def ABCD_plan(q_in, n1, n2):
    M = np.array([[1, 0], [0, n1 / n2]])
    q_out = np.matmul(M, q_in)
    if np.iscomplex(q_in[0]) == True:
        q_out = q_out / q_out[1]
    return(q_out)


def ABCD_curv(q_in, n1, n2, R):
    M = np.array([[1, 0], [(n1 - n2) / (R * n2), n1 / n2]])
    q_out = np.matmul(M, q_in)
    if np.iscomplex(q_in[0]) == True:
        q_out = q_out / q_out[1]
    return(q_out)
