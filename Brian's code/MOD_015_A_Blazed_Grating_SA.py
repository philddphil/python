#  MOD_015_A_Blazed_Grating_2.py
#
#  Code to calculate hologram using SA algorithm.
#
#  ROADMap Systems Ltd
#
#  Brian Robertson
#
#  08/Jan/2017
#
#  Version 1 - Basic code layout (26/Nov/2016).
#  Version 2 - Added scaling factors (27/Nov/2016).
#  Version 3 - Created consistent functions for calculation routines, replay
#              field, Gaussian incident beam, and edge effect (23/Dec/2016).
#  Version 4 - Added optimization algorithm (Python scipy minimize POWELL
#              function).  Added weighted merit term (30/12/2016).
#  Version 5 - Added SA algorithm, with blazed grating as starting point.
#              (08/01/2017)
#  Version 6 - Added passband calculation (final optimized CGH). (10/01/2017)

#******************************* HOUSEKEEPING ****************************
from numpy import*
from pylab import *
from math import*
import sys
from scipy.optimize import minimize
import time

#***************************** SET UP PARAMETERS *************************

wavelength = 1550e-9
clight = 3e8
N_pixels = 50
pixel = 6.4e-6
f = 200e-3
w = N_pixels * pixel / 5.0
period = float(10)
theta = asin(wavelength / (period * pixel))
pos_p1 = f * tan(theta)
pos_n1 = -f * tan(theta)

print('')
print('+1 order = ', pos_p1)
print('-1 order =', pos_n1)

pad = 8
res = 8
Np = N_pixels * res * pad

#***************************** DEFINE FUNCTIONS **************************


def replay_field(amp, phase_term, pad, res):

    count = size(phase_term)
    aa1 = res * count
    phase1 = zeros([aa1])

    for ii in range(aa1):
        kk = floor(ii / res)
        phase = phase_term[kk]
        phase1[ii] = phase

    aa2 = aa1 * pad
    phase2 = zeros([aa2])

    phase2[aa2 / 2 - aa1 / 2:aa2 / 2 + aa1 / 2] = phase1

    h1 = amp * np.exp(-1j * phase2)
    replay = fft(fftshift(h1))
    replay = fftshift(replay) / (aa2)
    intensity = (np.abs(replay))**2

    power = sum((np.abs(h1)**2) * (1 / pixel)) / (aa2 / pixel)
    power_freq_domain = sum(intensity)

    if (power - power_freq_domain) / power > 0.001:
        print('Potential error')

    return intensity, replay


def Gerchberg_Saxton(gin, CGH):

    levels = float(128)
    ite = 0
    M1 = 0
    M2 = -100

    while (M1 < 0.5) and M2 < -50:
        if ite == 0:
            ftg = fftshift(fft(fftshift(gin)))
        else:
            ftg = fftshift(fft(fftshift(gin * np.exp(-1j * CGH))))
            angle_ftg = np.angle(ftg)
            CGH = np.angle(
                fftshift(ifft(fftshift(target * np.exp(-1j * angle_ftg)))))

        phase_offset = pi + np.amin(CGH)

        for jj in range(N_pixels):
            q1 = floor(((CGH[jj] + pi) / (2 * pi)) * levels)
            q1 = q1 * 2 * pi / levels
            CGH[jj] = q1 - phase_offset

        M1, M2 = meritfn(CGH, gin, N_pixels, pad, res)
        ite = ite + 1

    return CGH


def meritfn(CGH, gin, N_pixels, pad, res):

    gin = zeros([Np])
    gin = Gaussian(gin, N_pixels, res, pad)
    text = 'GSA Replay field'
    replay_intensity, replay = replay_field(gin, CGH, pad, res, text)

    pos = 1560
    xt = 1640

    M1 = replay_intensity[pos] / max_I
    M2 = 10 * np.log10(replay_intensity[xt] / max_I)

    return M1, M2


def Gaussian(ga, N_pixels, res, pad):

    aa1 = N_pixels * res
    gtest = zeros([aa1])

    for ii in range(aa1):
        x = (aa1 / 2 - ii) * pixel / (1.0 * res)
        gtest[ii] = np.exp(-(x**2) / w**2)

    ga[Np / 2 - aa1 / 2:Np / 2 + aa1 / 2] = gtest

    return ga


def Gaussian2(ga, N_pixels, res, pad, offset):

    aa1 = N_pixels * res
    gtest = zeros([aa1])

    for ii in range(aa1):
        x = (aa1 / 2 - ii) * pixel / (1.0 * res)
        gtest[ii] = np.exp(-((x + offset)**2) / w**2)

    ga[Np / 2 - aa1 / 2:Np / 2 + aa1 / 2] = gtest

    return ga


def Edge(phase_term):

    count = size(phase_term)
    aa1 = res * count
    phase1 = zeros([aa1])

    for ii in range(aa1):
        kk = floor(ii / res)
        phase = phase_term[kk]
        phase1[ii] = phase

    aa2 = aa1 * pad
    phase2 = zeros([aa2])

    phase2[aa2 / 2 - aa1 / 2:aa2 / 2 + aa1 / 2] = phase1
    b = 0.004

    response = ones([1 + Np / 2])

    for ii in range(int(1 + Np / 2)):
        response[ii] = cos(-b * ii)
        if ii > int(pi / (2 * b)):
            response[ii] = 0.0

    Qfreq = rfft(phase2)
    Qfreq = Qfreq * response
    phase_term = irfft(Qfreq)

    return phase_term


def Grating2pi(phase_term, a0, a3, a5):

    count = size(phase_term)
    levels = float(128)

    for ii in range(count):
        p1 = 2 * pi * (ii / (period))
        p2 = floor(p1 / (2 * pi))
        p3 = p1 - p2 * 2 * pi
        if p3 > ((levels - 1 + 0.5) / levels) * 2 * pi:
            p3 = 0
        q1 = floor((p3 / (2 * pi)) * levels)
        q1 = q1 * 2 * pi / levels
        phase_term[ii] = q1 - pi

    for ii in range(count):
        p1 = phase_term[ii]
        p1 = a0 * p1 + a3 * p1**3 + a5 * p1**5
        phase_term[ii] = p1

    return phase_term


def performance(replay_intensity, max_I):

    signal = np.amax(replay_intensity / max_I)
    pos = np.argmax(replay_intensity)
    cp = Np / 2
    shift = cp - pos
    xt = cp + shift
    xt_signal = 10 * log10(replay_intensity[xt] / max_I)

    return signal, xt_signal, shift, cp, pos, xt


def crosstalk_calc(replay_ref, replay_CGH, shift):
    shift2 = -shift
    EsignalT = zeros([Np])
    EsignalT[Np / 2 - N_pixels * res / 2:Np / 2 + N_pixels * res / 2] = np.abs(replay_ref[Np / 2 - (
        N_pixels * res / 2) + shift2:Np / 2 + (N_pixels * res / 2) + shift2]) / sqrt(max_I)

    Efibre = EsignalT
    ESH = replay_CGH

    term1 = (sum((Efibre * abs(ESH))))**2
    term2 = sum(abs(ESH)**2)
    term3 = sum(abs(Efibre)**2)
    crosstalk = 10 * log10(abs(term1 / (term2 * term3)))

    return crosstalk, EsignalT


def function3(x):

    a0 = x[0]
    a3 = x[1]
    a5 = x[2]

    Np = N_pixels * res * pad
    CGH = zeros([N_pixels])
    CGH = Grating2pi(CGH, a0, a3, a5)
    phaseQ = zeros([Np])
    phaseQ = Edge(CGH)
    phaseQ = phaseQ
    gin = zeros([Np])
    gin = Gaussian(gin, N_pixels, res, pad)

    text = '2pi Replay field'
    pade = 1
    rese = 1
    replay_intensity, replay_CGH = replay_field(gin, phaseQ, pade, rese, text)

    signal, crosstalk, shift, cp, pos, xt = performance(
        replay_intensity, max_I)

    #crosstalk, EsignalT = crosstalk_calc(replay_ref, replay_CGH, shift)
    return crosstalk + 250 * np.log10(1 - signal)


def SA(CGH, error, error_vec, Iterations):
    # Set SA calculation parameters
    loop = 50
    na = 1.0
    steps = 128.0
    # Probability of accepting worse solution at the start
    p_start = 0.8
    # Probability of accepting worse solution at the end
    p_end = 0.001
    # Initial temperature
    t_start = -1.0 / np.log(p_start)
    # Final temperature
    t_end = -1.0 / np.log(p_end)
    # Fractional reduction every cycle
    frac = (t_end / t_start)**(1.0 / (Iterations - 1.0))
    # Set starting temperature
    T = t_start

    for ii in range(Iterations):
        # print ii
        for jj in range(loop):

            # Set random pixel and random phase
            phase_new = floor(np.random.rand(1) * (steps - 1)
                              ) * 2 * pi / steps - pi
            position = int(floor(np.random.rand(1) * N_pixels))
            phase_old = CGH[position]

            # Calculate new field
            CGH[position] = phase_new
            phaseQ = zeros([Np])
            phaseQ = Edge(CGH)
            intensity, replay = replay_field(gin, phaseQ, 1, 1)

            # Calculate replay error
            new_error = merit(intensity, replay, phaseQ, target)
            DeltaE = abs(new_error - error)
            if (ii == 0 and jj == 0):
                DeltaE_avg = DeltaE

            # SA algorithm
            if (new_error > error):
                p = np.exp(-DeltaE / (DeltaE_avg * T))
                if (np.random.rand(1) < p):
                    accept = True
                else:
                    accept = False
            else:
                accept = True

            if (accept == True):
                error = new_error
                na = na + 1.0
                DeltaE_avg = (DeltaE_avg * (na - 1.0) + DeltaE) / na
            else:
                CGH[position] = phase_old

        # Save current error value
        error_vec[ii] = error
        # Set new temperature
        T = frac * T

    return CGH, error_vec


def merit(intensity, replay, phaseQ, target):
    # +1 order diffraction efficiency
    replay_error_s = 100 * (1 - (intensity[pos] / target[pos]))
    # -1 order crosstalk
    replay_error_xt, EsignalT = crosstalk_calc(replay_ref, replay, shift)
    # Merit function term
    return (replay_error_s)**2 + abs(50 + replay_error_xt)**2


#***************************** FILE INFORMATION **************************

# File information
print ('')
print ('Program to optimise modulo 2pi phase profile using SA algorithm')
print ('')

print ('start calculation')
print (' ')
start = time.time()

#********************************** FLAT FIELD ***************************
# Calculate replay field when a flat field is applied and set up replay plane
# and CGH coordinate systems.

Np = N_pixels * res * pad
FP = zeros([N_pixels])
pixel_no = zeros([Np])
gin = zeros([Np])
gin = Gaussian(gin, N_pixels, res, pad)

replay_intensity_ref, replay_ref = replay_field(gin, FP, pad, res)
max_I = np.amax(replay_intensity_ref)

# Set replay plane spatial resolution
step2 = wavelength * f / Np / pixel
xoutFP = zeros([Np])
for ii in range(Np):
    xoutFP[ii] = (Np / 2 - ii) * step2

# Set up CGH plane spatial coordinates
for ii in range(Np):
    pixel_no[ii] = ii * 1 / (1.0 * res)

c_pixel = Np / (1.0 * res) / 2.0

#********************************* 2PI GRATING ***************************
# Calculate initial starting point using modulo(2pi) algorithm

Np = N_pixels * res * pad
CGH = zeros([N_pixels])

# Polynomial phase terms (set as linear)
a0 = 1
a3 = 0
a5 = 0

CGH = Grating2pi(CGH, a0, a3, a5)
pad = 8
res = 8
phaseQ = zeros([Np])
phaseQ = Edge(CGH)
gin = zeros([Np])
gin = Gaussian(gin, N_pixels, res, pad)

intensity, replay = replay_field(gin, phaseQ, 1, 1)
signal, crosstalk, shift, cp, pos, xt = performance(intensity, max_I)
crosstalk, EsignalT = crosstalk_calc(replay_ref, replay, shift)

print ('Initial diffraction efficiency = ', signal * 100, '%')
print ('Initial crosstalk = ', crosstalk, 'dB')
print ('')
print ('Minimum phase = ', np.amin(phaseQ))
print ('Maximum phase =', np.amax(phaseQ))
print ('Phase range =', (np.amax(phaseQ) - np.amin(phaseQ)) / (pi))
print ('')

signal_complex = replay[pos]
crosstalk_complex = replay[xt]
relative_field = abs(signal_complex) / abs(crosstalk_complex)
print ('Ratio of -1 to +1 = ', relative_field)
angle_signal = angle(signal_complex)
print ('Signal angle = ', angle_signal * 180 / pi)
angle_crosstalk = angle(crosstalk_complex)
print ('Crosstalk angle =', angle_crosstalk * 180 / pi)
print ('')

#************************************* SA ********************************
# Using blazed grating as starting point, apply a simulated annealing algorithm
# to maintain diffraction efficiency, whilst suppressing crosstalk in -1 order.

target = zeros([Np])
target[pos] = max_I
target[xt] = 0

# Calculate initial error
error = merit(intensity, replay, phaseQ, target)
print ('Error initial = ', error)
print ('')

pad = 8
res = 8

# Core SA algorithm
Iterations = 500
error_vec = zeros([Iterations])
CGH, error_vec = SA(CGH, error, error_vec, Iterations)

# Calculate optimised replay field
pad = 8
res = 8
phaseQ = zeros([Np])
phaseQ = Edge(CGH)
gin = zeros([Np])
gin = Gaussian(gin, N_pixels, res, pad)
replay_intensity, replay = replay_field(gin, phaseQ, 1, 1)

signal, crosstalk, shift2, cp2, pos2, xt2 = performance(
    replay_intensity, max_I)
crosstalk, EsignalT = crosstalk_calc(replay_ref, replay, shift)

#**************************** GRATING PERFORMANCE ************************

print ('Final diffraction efficiency = ', signal * 100, '%')
print ('Crosstalk final = ', crosstalk, 'dB')
print ('')
print ('Minimum phase = ', np.amin(phaseQ))
print ('Maximum phase =', np.amax(phaseQ))
print ('Phase range =', (np.amax(phaseQ) - np.amin(phaseQ)) / (pi))
print ('')
signal_complex = replay[pos]
crosstalk_complex = replay[xt]
relative_field = abs(signal_complex) / abs(crosstalk_complex)
print ('Ratio of -1 to +1 = ', relative_field)
angle_signal = angle(signal_complex)
print ('Signal angle = ', angle_signal * 180 / pi)
angle_crosstalk = angle(crosstalk_complex)
print ('Crosstalk angle =', angle_crosstalk * 180 / pi)
print ('')

end = time.time()
print ('time for optimisation = ', (end - start), 'seconds')
print ('')

xt_pos = [xt, xt]
sig_pos = [pos, pos]
zeroth_pos = [cp, cp]
limits = [-50, 0]

#*************************** PASSBAND MAIN LOOP **************************
step = 50
power = zeros([step + 1])
xtalk_power = zeros([step + 1])
res = 8
pad = 8
pade = 1
rese = 1
text = 'Test'

gin = zeros([Np])

for count in range(step + 1):

    y_offset = ((-N_pixels / 2.0) + count) * pixel
    gin = Gaussian2(gin, N_pixels, res, pad, y_offset)

    intensity_passband, replay = replay_field(gin, phaseQ, pade, rese)
    signal, crosstalk2, shift2, cp2, pos2, xt2 = performance(
        intensity_passband, max_I)
    crosstalk, EsignalT = crosstalk_calc(replay_ref, replay, shift)

    xtalk_power[count] = crosstalk
    power[count] = 10 * np.log10(np.amax(signal))


max_power = np.amax(power)
print ('Min insertion loss = ', max_power)
passband_limit = max_power - 0.5

xt = [0, 50]
yt = [passband_limit, passband_limit]


#******************************* PLOT FIGURES ****************************

plt.subplot(221)
plot(pixel_no - c_pixel + floor(N_pixels / 2), phaseQ / (pi), 'b')
#plot(pixel_no-c_pixel+floor(N_pixels/2), 6*(abs(gin))**2, 'r')
xlabel('CGH plane (pixels)')
ylabel('Optimised phase (Radians)')
axis([0, N_pixels - 1, -2 * pi, 2 * pi])

plt.subplot(222)
plot(xoutFP, replay_intensity / max_I, 'b')
plot(xoutFP, replay_intensity_ref / max_I, 'r')
xlabel('Position in replay field')
ylabel('Normalized intensity')

plt.subplot(223)
plot(10 * np.log10(replay_intensity / max_I), 'b')
plot(zeroth_pos, limits, 'r')
plot(xt_pos, limits, 'g')
plot(sig_pos, limits, 'b')
xlabel('Position in replay field')
ylabel('Normalized intensity (10log10)')
axis([Np / 2 - 500, Np / 2 + 500, -50, 0])

figure(2)
plot(error_vec)
xlabel('Number of iterations')
ylabel('Error')

figure(3)
plot(CGH)

#****************************** PLOT RESULTS *****************************

figure(4)
plot(power)
plot(xt, yt)
xlabel('Frequency (GHz)')
ylabel('Insertion loss (dB)')

figure(5)
plot(xtalk_power)
xlabel('Frequency (GHz)')
ylabel('Crosstalk (dB)')

show()
