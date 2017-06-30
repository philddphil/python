import matplotlib.pyplot as plt
import random
import os
import glob
import numpy as np
import scipy.io as io
import pylab as pl
import scipy.optimize as opt

from scipy.interpolate import interp1d


def Phase(x, A, B):
    y = np.square(np.sin(A * (1 - np.exp(-B * x))))
    return y


p1 = r'C:\Users\Philip\Documents\Python files\curve fitting'
os.chdir(p1)

files = glob.glob('*Phaseramp.mat')

for i1 in range(len(files)):

    phaseramp = io.loadmat(files[i1])
    print(files[i1])
    y_dB1 = phaseramp['P4'].ravel()
    y_dB2 = phaseramp['P6'].ravel()
    y_lin1 = np.power(10, y_dB1 / 10) / np.max(np.power(10, y_dB1 / 10))
    y_lin2 = np.power(10, y_dB2 / 10) / np.max(np.power(10, y_dB2 / 10))
    x0 = np.linspace(0, 255, len(y_dB1))

    x1 = np.linspace(0, 255, 25)
    f1 = interp1d(x0, y_lin1)
    f2 = interp1d(x0, y_lin2)
    plt1 = pl.figure('Phaseramp plot 1')
    pl.plot(x1, f1(x1), 'ob')
    pl.plot(x1, f2(x1), 'or')
    pl.plot(x0, y_lin1, 'b')
    pl.plot(x0, y_lin2, 'r')

    initial_guess = (15, 1 / 800)
    x2 = range(255)
    try:
        popt1, pcov = opt.curve_fit(Phase, x1, f1(x1), p0=initial_guess)
        y_test1 = Phase(x2, *popt1)
        pl.plot(x2, y_test1, 'b')
    except RuntimeError:
        print("Error - curve_fit failed")

    try:
        popt2, pcov = opt.curve_fit(Phase, x1, f2(x1), p0=initial_guess)
        y_test2 = Phase(x2, *popt2)
        pl.plot(x2, y_test2, 'r')
    except RuntimeError:
        print("Error - curve_fit failed")

    pl.title('Γρεεκ Λεττερσ ρυλε '+ files[i1])

    pl.show()
