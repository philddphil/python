##############################################################################
# Import some libraries
##############################################################################

import os
import glob
import copy
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as io
import scipy.optimize as opt

from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter
from PIL import Image

###############################################################################
# Define some functions
###############################################################################


# Generate holograms with first two parameters to optimise - Λ and φ ##########
def holo_tilt(Λ, φ, Hol_δy, Hol_δx, ϕ_min, ϕ_max):

    x = np.arange(Hol_δx)
    y = np.arange(Hol_δy)
    [X, Y] = np.meshgrid(x, y)

    θ = np.arctan((ϕ_max - ϕ_min) / Λ)

    Z = np.tan(θ) * (X * np.cos(φ) + Y * np.sin(φ))
    Z_mod = Z % (ϕ_max - ϕ_min - 0.00000001)
    Z_mod = Z_mod * (ϕ_max - ϕ_min) / (np.max(Z_mod)) + ϕ_min

    Holo_s = (Z, Z_mod)
    return Holo_s


# Add sub hologram Z_mod to larger hologram (initially set to 0s) #############
def add_holo(Hol_cy, Hol_cx, Z_mod, LCOSy, LCOSx):
    LCOSy = int(LCOSy)
    LCOSx = int(LCOSx)

    Holo_f = np.zeros((LCOSy, LCOSx))
    (Hol_δy, Hol_δx) = np.shape(Z_mod)
    y1 = np.int(Hol_cy - np.floor(Hol_δy / 2))
    y2 = np.int(Hol_cy + np.ceil(Hol_δy / 2))
    x1 = np.int(Hol_cx - np.floor(Hol_δx / 2))
    x2 = np.int(Hol_cx + np.ceil(Hol_δx / 2))
    Holo_f[y1:y2, x1:x2] = Z_mod
    return Holo_f


# Defining the functional form of grayscale to phase (g(ϕ)) ###################
def phase(x, A, B):
    ϕ = np.square(np.sin(A * (1 - np.exp(-B * x))))
    return ϕ


# Use g(ϕ) defined in 'phase' to fit experimentally obtained phaseramps #######
def fit_phase():
    f1 = r'C:\Users\Philip\Documents\LabVIEW\Data\Calibration files\Phaseramp.mat'
    # f1 = r'..\..\Data\Calibration files\*Phaseramp.mat'
    files = glob.glob(f1)
    phaseramp = io.loadmat(files[0])

    y_dB = phaseramp['P4'].ravel()
    y_lin = np.power(10, y_dB / 10) / np.max(np.power(10, y_dB / 10))

    x0 = np.linspace(0, 255, len(y_dB))
    x1 = np.linspace(0, 255, 25)
    x3 = range(255)
    f1 = interp1d(x0, y_lin)
    initial_guess = (15, 1 / 800)

    try:
        popt, pcov = opt.curve_fit(phase, x1, f1(
            x1), p0=initial_guess, bounds=([0, -np.inf], [np.inf, np.inf]))

    except RuntimeError:
        print("Error - curve_fit failed")

    ϕ_A = popt[0]
    ϕ_B = popt[1]
    ϕ_g = (2 / np.pi) * np.abs(ϕ_A) * (1 - np.exp(-ϕ_B * x3))

    return (ϕ_A, ϕ_B, ϕ_g)


# Use the fitting results from 'fit_phase'  to remap hologram Z_mod ###########
def remap_phase(Z_mod, g_ϕ):
    Z_mod1 = copy.copy(Z_mod)
    for i1 in range(np.shape(Z_mod)[0]):
        Z_mod1[i1, :] = g_ϕ(Z_mod[i1, :])
    return (Z_mod1)


# Save a hologram (nupmy array) to a .bmp #####################################
def save_bmp(Hologram, Path):
    plt.imsave(Path + '.png', Hologram, cmap=plt.cm.gray, vmin=0, vmax=255)
    file_in = Path + '.png'
    img = Image.open(file_in)
    file_out = Path + '.bmp'
    img.save(file_out)


# Overshoot mapping ###########################################################
def overshoot_phase(Z_mod1, g_OSlw, g_OSup, g_min, g_max):
    Z_mod2 = copy.copy(Z_mod1)
    Super_thres_indices = Z_mod1 > g_OSup
    Sub_thres_indices = Z_mod1 <= g_OSlw
    Z_mod2[Super_thres_indices] = g_max
    Z_mod2[Sub_thres_indices] = g_min
    return (Z_mod2)


# Upack values from Hologram control sent by LabVIEW ##########################
def variable_unpack(LabVIEW_data):
    LCOS_δx = LabVIEW_data[0]
    LCOS_δy = LabVIEW_data[1]

    Hol_δx = LabVIEW_data[2]
    Hol_δy = LabVIEW_data[3]
    Hol_cx = LabVIEW_data[4]
    Hol_cy = LabVIEW_data[5]

    ϕ_min = LabVIEW_data[6]
    ϕ_max = LabVIEW_data[7]
    ϕ_lwlim = LabVIEW_data[8]
    ϕ_uplim = LabVIEW_data[9]

    g_OSlw = LabVIEW_data[10]
    g_OSup = LabVIEW_data[11]
    g_min = LabVIEW_data[12]
    g_max = LabVIEW_data[13]

    Λ = LabVIEW_data[14]
    φ = LabVIEW_data[15]

    params = [LCOS_δx, LCOS_δy,
              Hol_δx, Hol_δy, Hol_cx, Hol_cy,
              ϕ_min, ϕ_max, ϕ_lwlim, ϕ_uplim,
              g_OSlw, g_OSup, g_min, g_max,
              Λ, φ]
    return params


# Generate hologram and save as bmp ###########################################
def holo_gen(*LabVIEW_data):
    # Unpack parameters
    LCOS_δx = LabVIEW_data[0]
    LCOS_δy = LabVIEW_data[1]

    Hol_δx = LabVIEW_data[2]
    Hol_δy = LabVIEW_data[3]
    Hol_cx = LabVIEW_data[4]
    Hol_cy = LabVIEW_data[5]

    ϕ_lwlim = LabVIEW_data[8]
    ϕ_uplim = LabVIEW_data[9]

    g_OSlw = LabVIEW_data[10]
    g_OSup = LabVIEW_data[11]
    g_min = LabVIEW_data[12]
    g_max = LabVIEW_data[13]

    Λ = LabVIEW_data[14]
    φ = LabVIEW_data[15]

    # Phase mapping details (ϕ)
    (ϕ_A, ϕ_B, ϕ_g) = fit_phase()
    g_ϕ = interp1d(ϕ_g, range(255))

    # Define holo params
    LCOS_δyx = (LCOS_δy, LCOS_δx)
    Hol_δyx = (Hol_δy, Hol_δx)
    Hol_cyx = (Hol_cy, Hol_cx)
    ϕ_lims = (ϕ_lwlim, ϕ_uplim)
    Holo_params = (Λ, φ, *Hol_δyx, *ϕ_lims)

    # Calculate sub hologram (Holo_s)
    Holo_s = holo_tilt(*Holo_params)
    Zϕ_mod = Holo_s[1]

    # Remap phase with non linear ϕ map
    Zg_mod1 = remap_phase(Zϕ_mod, g_ϕ)

    # Use overshooting
    Zg_mod2 = overshoot_phase(Zg_mod1, g_OSlw, g_OSup, g_min, g_max)

    # Calculate full holograms (Holo_f)
    Holo_f2 = add_holo(*Hol_cyx, Zg_mod2, *LCOS_δyx)

    # Set output holograms (Z_out, Holo_out)
    Holo_out = Holo_f2
    # Save output
    save_bmp(Holo_out, r'..\..\Data\bmps\hologram')


# Generate 'phase mapping image' for LabVIEW FP ###############################
def phase_plot(*LabVIEW_data):
    # Unpack parameters
    ϕ_lwlim = LabVIEW_data[8]
    ϕ_uplim = LabVIEW_data[9]

    (ϕ_A, ϕ_B, ϕ_g) = fit_phase()
    g_ϕ = interp1d(ϕ_g, range(255))
    ϕ_min = min(ϕ_g)
    ϕ_max = max(ϕ_g)

    plt.plot(ϕ_g, '.-', Color='xkcd:blue')
    plt.plot(ϕ_min * np.ones(255), Color='xkcd:light red')
    plt.plot(ϕ_max * np.ones(255), Color='xkcd:light red')

    plt.plot(ϕ_lwlim * np.ones(255), ':', Color='xkcd:light red')
    plt.plot(ϕ_uplim * np.ones(255), ':', Color='xkcd:light red')

    plt.axvline(x=g_ϕ(ϕ_lwlim), Color='xkcd:light blue')
    plt.axvline(x=g_ϕ(ϕ_uplim), Color='xkcd:light blue')

    plt.text(180, ϕ_max + 0.2, 'ϕ$_{max}$ = ' + str(np.round(ϕ_max, 3)),
             horizontalalignment='left', size=20)
    os.remove(r'..\..\Data\bmps\phase.png')
    plt.savefig(r'..\..\Data\bmps\phase.png')
    plt.cla()
    return (ϕ_min, ϕ_max, g_ϕ)


# Generic 1D Gaussian function ################################################
def Gaussian_1D(x, A, xo, σ_x, bkg):
    xo = float(xo)
    g = bkg + A * np.exp(- ((x - xo) ** 2) / (2 * σ_x ** 2))
    return g


# Generic 2D Gaussian function ################################################
def Gaussian_2D(coords, A, xo, yo, σ_x, σ_y, θ, bkg):
    x, y = coords
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(θ) ** 2) / (2 * σ_x ** 2) + (np.sin(θ) ** 2) / (2 * σ_y ** 2)
    b = -(np.sin(2 * θ)) / (4 * σ_x ** 2) + (np.sin(2 * θ)) / (4 * σ_y ** 2)
    c = (np.sin(θ) ** 2) / (2 * σ_x ** 2) + (np.cos(θ) ** 2) / (2 * σ_y ** 2)
    g = (bkg + A * np.exp(- (a * ((x - xo) ** 2) +
                             2 * b * (x - xo) * (y - yo) +
                             c * ((y - yo) ** 2))))
    return g.ravel()


# Fit Λ and ϕ datasets from peak finding routine ##############################
def find_fit_peak(x, y, A, xo):
    x_1 = np.linspace(min(x), max(x), 100)
    Peak_ind = np.unravel_index(y.argmax(), y.shape)
    initial_guess = (A, x[Peak_ind[0]], xo, 0)

    # Plot data
    # plt.figure(1)
    # plt.plot(x, y, c='xkcd:light red', marker='.')
    # plt.plot(x, Gaussian_1D(x, *initial_guess), c='xkcd:light blue', marker='.')
    # plt.show()
    # Fit data
    try:
        popt, pcov = opt.curve_fit(
            Gaussian_1D, x, y, p0=initial_guess)
        fit0 = Gaussian_1D(x_1, *popt)

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
    coords = (X, Y)
    X1, Y1 = np.meshgrid(x1, y1)
    coords1 = (X1, Y1)

    RBS_f = RectBivariateSpline(y, x, im)
    RBS_im = RBS_f(y1, x1)
    G_RBS_im = gaussian_filter(RBS_im, 10)
    G_RBS = RectBivariateSpline(y1, x1, G_RBS_im)
    smooth_im = G_RBS(y, x)
    return smooth_im


# Gaussian blur an image n times ##############################################
def n_G_blurs(im, n):
    im_out = im
    for i1 in range(n):
        im_out = gaussian_filter(im_out, 20)

    return im_out


# Find the RMS of array a #####################################################
def rms(a):
    b = (np.sqrt(np.mean(np.square(a))))
    return b


# Find the RMS of array a #####################################################
def img_csv(file):
    im = np.genfromtxt(file, delimiter=',')
    im_size = np.shape(im)
    y = np.arange(im_size[0])
    x = np.arange(im_size[1])
    X, Y = np.meshgrid(x, y)
    coords = (X, Y)
    return (im, coords)


# Return the element location of the max of array a ###########################
def max_i_2d(a):
    b = np.unravel_index(a.argmax(), a.shape)
    return(b)


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
               'rmp_dblue': [12 / 255, 35 / 255, 218 / 255],
               'rmp_lblue': [46 / 94, 38 / 249, 86 / 255],
               'rmp_pink': [210 / 255, 76 / 255, 197 / 255],
               'rmp_green': [90 / 255, 166 / 255, 60 / 255]}

    plt.style.use('ggplot')
    plt.rcParams['font.size'] = 8
    plt.rcParams['font.family'] = 'monospace'
    plt.rcParams['font.fantasy'] = 'Nimbus Mono'
    plt.rcParams['axes.labelsize'] = 8
    plt.rcParams['axes.labelweight'] = 'normal'
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 10
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
