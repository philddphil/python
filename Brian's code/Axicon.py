#  Linear phase mask.py
#
#  Code to calculate hologram using Gerchberg-Saxton algorithm.
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
#  Version 4 - Update to add mode-overlap calculaiton.
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

#***************************** SET UP PARAMETERS *************************

wavelength = 1550e-9
clight = 3e8
N_pixels = 50
pixel = 6.4e-6
w = N_pixels * pixel / 5.0
wf = 5.2e-6
f = wf * pi * w / wavelength
zp = f
p = 10
theta_m = np.arcsin(wavelength / (p * pixel))
yp = f * tan(theta_m)
xp = f * tan(theta_m)

# Define axicon wedge angle

beta = 0 * pi / 180.0
beta2 = -1 * beta

# Definition of propagation vector

v_mag = np.sqrt(f**2 + xp**2 + yp**2)
nx = xp / v_mag
ny = yp / v_mag
nz = zp / v_mag

# Number of phase levels

levels = 128.0

#***************************** DEFINE FUNCTIONS **************************


def replay_field(amp, phase_term, pad, res, text):

    aa1 = N_pixels * res * pad

    h1 = amp * np.exp(-1j * phase_term)
    replay = np.fft.fft2(fftshift(h1))
    replay = fftshift(replay) / (aa1**2)
    intensity = (np.abs(replay))**2

    power = sum((np.abs(h1)**2) * (1 / pixel**2)) / (aa1**2 / pixel**2)
    power_freq_domain = sum(intensity)

    if (power - power_freq_domain) / power > 0.001:
        print('Potential error')

    return intensity, replay


def Gaussian(ga, N_pixels, res, pad):

    aa1 = N_pixels * res
    gtest = zeros([aa1, aa1])

    for ii in range(aa1):
        for jj in range(aa1):
            x = (aa1 / 2 - ii) * pixel / (1.0 * res)
            y = (aa1 / 2 - jj) * pixel / (1.0 * res)
            gtest[ii, jj] = np.exp(-(x**2 + y**2) / w**2)

    ga[Np / 2 - aa1 / 2:Np / 2 + aa1 / 2, Np /
        2 - aa1 / 2:Np / 2 + aa1 / 2] = gtest

    return ga


def Gaussian2(ga, N_pixels, res, pad, offset):

    aa1 = N_pixels * res
    gtest = zeros([aa1, aa1])

    for ii in range(aa1):
        for jj in range(aa1):
            x = (aa1 / 2 - ii) * pixel / (1.0 * res)
            y = (aa1 / 2 - jj) * pixel / (1.0 * res)
            gtest[ii, jj] = np.exp(-((x + offset)**2 + y**2) / w**2)

    ga[Np / 2 - aa1 / 2:Np / 2 + aa1 / 2, Np /
        2 - aa1 / 2:Np / 2 + aa1 / 2] = gtest

    return ga


def transmitted(offset, wbeam, pixel_size, N_pixels):

    aa = wbeam / sqrt(2)
    a = pixel_size * N_pixels / 2.0
    b = pixel_size * N_pixels / 2.0

    reference_power = (erf(a / aa) - erf(-a / aa)) * \
        (erf(b / aa) - (erf(-b / aa)))

    Tx = erf((a - offset) / aa) - erf((-a - offset) / aa)
    Ty = erf(b / aa) - erf(-b / aa)
    return Tx * Ty / reference_power


def crosstalk_calc(ref_field, replay_CGH, shift):
    shift2 = -shift
    Efibre = zeros([Np])
    Efibre[Np / 2 - N_pixels * res / 2:Np / 2 + N_pixels * res / 2] = np.abs(ref_field[Np / 2 - (
        N_pixels * res / 2) + shift2:Np / 2 + (N_pixels * res / 2) + shift2]) / np.sqrt(max_I)
    term1 = (sum((Efibre * abs(replay_CGH))))**2
    term2 = sum(abs(replay_CGH)**2)
    term3 = sum(abs(Efibre)**2)
    crosstalk = abs(term1 / (term2 * term3))

    return crosstalk, Efibre


#***************************** FILE INFORMATION **************************

# File information

print ('Program to analyse beam profile')


print ('start calculation')

start = time.time()

#********************************** MAIN LOOP ****************************

#  Set-up phase profile

px = N_pixels
py = N_pixels

phase_SLM = zeros([px, py])
grey_SLM = zeros([px, py])

for n in range(py):
    for m in range(px):
        xSLM = n * pixel
        ySLM = m * pixel
        h = (nx * xSLM + ny * ySLM) / nz

        # Calculate conical phase profile

        xA = (-px / 2 + n + 0.5) * pixel
        yA = (-py / 2 + m + 0.5) * pixel
        radius = np.sqrt(xA**2 + yA**2)
        h1 = radius * tan(beta)
        # Add two phase profiles
        h = h + h1
        # Corresponding phase step
        phase = (2 * pi * (h / wavelength))
        # Phase step according to modulo 2pi algorithm
        diff_phase = 2 * pi * ((phase / (2 * pi)) - floor(phase / (2 * pi)))
        # Quantize phase according to number of available phase levels
        q1 = floor((diff_phase / (2 * pi) * levels))
        quant_phase = q1 * 2 * pi / levels
        # Convert to bit-map value (scaled between 0 and 255)
        q2 = round(255 * q1 / levels)
        # Output arrays for phase delay and grey level
        phase_SLM[m, n] = quant_phase
        grey_SLM[m, n] = q2

imshow(phase_SLM)
show()

# Increase resolution

# Increase calculation resolution by representing each pixel by NxN
# elements to give a total phase and amplitude field of NTxNT points
N = 8
NT = px * N

Np = zeros([NT, NT])
Amp_Np = zeros([NT, NT])

# Set central phase value in each pixel
for ii in range(NT):
    for jj in range(NT):
        if ((ii / N) == floor(ii / N)) and ((jj / N) == floor(jj / N)):
            Np[ii - N + 1 + N / 2, jj - N + 1 +
                N / 2] = phase_SLM[ii / N, jj / N]


# Redifine px and py as equaling NT (new number of phase and amplitude points)
px = NT
py = NT

# PSF generation

# Calculate phase profile for NTxNT points by convolving Np with the psf
# phase_SLM = conv2(Np,pixel_define);

# Clip phase_SLM so it contains NTxNT points
phase_SLM = Np

# Define a point-spread function representing each pixel
psf = zeros([NT, NT])

NN = 2.0
w_edge = 1.0e-6

for ii in range(NT):
    for jj in range(NT):
        xpos = ((-NT / 2) + 0.5 + ii) * pixel / N
        ypos = ((-NT / 2) + 0.5 + jj) * pixel / N
        rpos = np.sqrt(xpos**2 + ypos**2)
        psf[ii, jj] = 1 * np.exp(-(rpos)**2 / (2 * (w_edge)**2))**NN
        #psf[ii,jj] = 0;
        # if (abs(xpos) < pixel/2) && (abs(ypos) < pixel/2)
        # psf(ii,jj) = 1;

max_psf = np.max(np.max(psf))
psf = psf / max_psf
imshow(psf)
show()

# Define beam amplitude profile

# Define incident beam amplitude distribution
SLM_amplt = zeros([px, py])

for ii in range(px):
    for jj in range(py):
        xSLM = (-px / 2 + 0.5 + ii) * pixel / N
        ySLM = (-py / 2 + 0.5 + jj) * pixel / N
        amplitude = np.exp(-1 * (xSLM**2 + ySLM**2) / (w**2))
        SLM_amplt[ii, jj] = amplitude

imshow(SLM_amplt)
show()

# Define new phase profile
phase_SLM_1 = fftshift(fft2(fftshift(phase_SLM))) * \
    fftshift(fft2(fftshift(psf)))
phase_SLM_2 = fftshift(ifft2(fftshift(phase_SLM_1)))

max_phase = np.max(np.max(phase_SLM_2))
min_phase = np.min(np.min(phase_SLM_2))
phase_SLM_2 = phase_SLM_2 * (2 * pi / (max_phase))

plot(phase_SLM_2[px / 2, :])

# Lock

phase_SLM_L = zeros([NT, NT])

for n in range(py):
    for m in range(px):
        # Calculate wavefront ramp, h, at location (xp, yp)

        # Calculate conical phase profile
        xA = (-px / 2 + 0.5 + n - 1) * pixel / N
        yA = (-py / 2 + 0.5 + m - 1) * pixel / N
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

        # Output arrays for phase delay and grey level
        phase_SLM_L[n, m] = quant_phase_L

imshow(phase_SLM_L)
show()

# Main calculation

# Define phase distribution when there is no hologram displayed
SLM_zero = zeros([px, py])

# Define zero padding factor, pad, and generate associated replay field
# calaculation matrices
pad = 4
E_calc = zeros([pad * px, pad * py])
E_calc_phase = zeros([pad * px, pad * py]) * 0j
E_calc_amplt = zeros([pad * px, pad * py]) * 0j
E_replay_zero = zeros([pad * px, pad * py]) * 0j

# Calculation of replay field when no grating is displayed
E_calc_phase[0:px, 0:px] = SLM_zero[:, :]
E_calc_amplt[0:px, 0:px] = SLM_amplt[:, :]
E_replay = fftshift(fft2(fftshift(E_calc_amplt * np.exp(1j * E_calc_phase))))
E_replay_zero = E_replay
I_replay_zero = (abs(E_replay))**2
# Maximum intensity
I_max_zero = np.max(np.max(I_replay_zero))
# Normalized replay field
I_replay_zero = I_replay_zero / I_max_zero


# Calculation of replay field when grating is displayed
E_calc_phase[0:px, 0:px] = phase_SLM_2[:, :] + phase_SLM_L[:, :]
E_calc_amplt[0:px, 0:px] = SLM_amplt[:, :]
E_replay = fftshift(fft2(fftshift(E_calc_amplt * np.exp(1j * E_calc_phase))))
I_replay = (abs(E_replay))**2
# Maximum intensity
I_max_signal = np.max(np.max(I_replay))
# Replay intensity distribution normalized with respect to the undiffracted
# zeroth order
I_replay = I_replay / I_max_zero

# Corresponding insertion loss
Loss = I_max_signal / I_max_zero
print (Loss)

# Calculate scaling vector and plot

x_vec = zeros([NT * pad])
# Calculate replay field scaled distance (in microns)
for ii in range(NT * pad):
    pos = ((1 + NT * pad / 2) - ii) * f * wavelength / ((pixel / N) * pad * NT)
    x_vec[ii] = pos * 1e6


# Calculate Mode-overlap integral

wrange = 100

[xpos, ypos] = np.where(I_replay_zero == I_replay_zero.max())
Replay_zero = abs(
    E_replay_zero[xpos - wrange:xpos + wrange, ypos - wrange:ypos + wrange])

[xpos, ypos] = np.where(I_replay == I_replay.max())
Replay_sig = abs(E_replay[xpos - wrange:xpos +
                          wrange, ypos - wrange:ypos + wrange])

points = 2 * wrange

term1 = 0
term2 = 0
term3 = 0

for mm in range(points):
    for nn in range(points):
        E_R = Replay_sig[mm, nn]
        E_F = Replay_zero[mm, nn]
        dterm1 = (abs(E_R))**2
        dterm2 = (abs(E_F))**2
        dterm3 = abs(abs(E_R) * abs(E_F))
        term1 = term1 + dterm1
        term2 = term2 + dterm2
        term3 = term3 + dterm3


eff = ((term3)**2) / ((term1) * (term2))
eff_log = 10 * np.log10(eff)
print (eff)


# Alternative calculation
test = (np.sum((np.abs(Replay_sig) * np.abs(Replay_zero))))**2 / \
    ((np.sum(np.abs(Replay_sig)**2)) * (np.sum(np.abs(Replay_zero)**2)))
print ('test = ', test)


figure(2)
plot(I_replay)

I1_plot = zeros([pad * px, pad * py])
I1_plot = I_replay_zero
I1_final = zeros([200, 200])
I1_final = 10 * np.log10(I1_plot[NT * pad / 2 - 100:NT *
                                 pad / 2 + 100, NT * pad / 2 - 100:NT * pad / 2 + 100])
#I1_plot = (I1_plot * 255).astype(np.uint8)
I1_final[I1_final < -50] = -50

I2_plot = zeros([pad * px, pad * py])
I2_plot = I_replay
I2_final = zeros([200, 200])
I2_final = 10 * np.log10(I2_plot[NT * pad / 2 - 100:NT *
                                 pad / 2 + 100, NT * pad / 2 - 100:NT * pad / 2 + 100])
I2_final[0, 0] = 0
I2_final[I2_final < -50] = -50


figure(3)
plt.subplot(221)
imshow(I1_final, cmap='gray')
plt.colorbar()
plt.subplot(222)
imshow(I2_final, cmap='gray')
plt.colorbar()

figure(4)
plt.subplot(221)
imshow(I1_final)
plt.colorbar()
plt.subplot(222)
imshow(I2_final)
plt.colorbar()


show()
