##############################################################################
# Import some libraries
##############################################################################
import sys
import os
import glob
import time
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import csv
import scipy.optimize as opt
import socket
import scipy.io as io
import importlib.util
import ntpath

from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import interp1d
from scipy.signal import find_peaks_cwt
from scipy.ndimage.filters import gaussian_filter
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import savgol_filter


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
p1 = (r"C:\Users\Philip\Documents\Technical Stuff\Hologram optimisation"
      r"\Algorithmic implementation\180227\Post realignment\Port 6")


print(p1)
data1 = prd.load_multicsv(p1)

print(np.shape(data1), np.shape(data1))
os.chdir(p1)
f1 = p1 + r'\*00.csv'
files = glob.glob(f1)
files = [files[i] for i in [3, 4]]
# files = [files[i] for i in [1, 2, 3, 4, 5, 6]]
# name = 'Spectra others'
# name = 'Spectra all'
name = 'Spectra +1,  -1 - small'

print(files)
data_all = np.array([])

fig1 = plt.figure('fig1', figsize=(3, 2.5))
# fig1 = plt.figure('fig1')
ax1 = fig1.add_subplot(1, 1, 1)
fig1.patch.set_facecolor(cs['mdk_dgrey'])
ax1.set_xlabel('Wavelength (μm)')
ax1.set_ylabel('Insertion Loss (dB)')

for i1, val1 in enumerate(files[:]):
    data = np.genfromtxt(val1, delimiter=',')
    number = re.findall(r'[-+]?\d+[\.]?\d*', val1)
    fibre = str(int(np.round(float(number[-1]))))
    _, idx1 = prd.find_nearest(data[0, :], 1.550)
    _, idx2 = prd.find_nearest(data[0, :], 1.552)
    print(idx1, idx2)
    data_ROI = data[:, idx1:idx2]
    avg1 = np.mean(data_ROI[1, :])
    data_ROI = data[:, idx1:idx2]
    avg1 = np.mean(data_ROI[1, :])
    label = fibre + ' ' + str(np.round(avg1, 2))
    fibre_c = 'fibre9d_' + fibre
    print(label)
    print(np.shape(data))
    plt.plot(data[0, :], data[1, :], '-', lw=1.5, label=label, c=cs[fibre_c])
    print(avg1)
    plt.plot(data_ROI[0, :], avg1 * np.ones(idx2 - idx1),
             '-', lw=1.5, c=cs['ggred'])
    # plt.fill_between(data_ROI[0, :], data_ROI[1, :], -
    #                  70 * np.ones(idx2 - idx1), lw=1, color=cs[fibre_c])

leg1 = plt.legend(prop={'size': 7})
leg1.get_frame().set_alpha(0.5)
# leg1.get_frame().set_facecolor(cs['mdk_dgrey'])
for text in leg1.get_texts():
    text.set_color("black")

##############################################################################
# Plot some figures
##############################################################################
os.chdir(p1)
ax1.set_aspect('auto')
plt.tight_layout()
plt.show()

prd.PPT_save_2d(fig1, ax1, name)
