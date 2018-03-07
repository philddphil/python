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
p1 = r"C:\Users\Philip\Documents\Data\Insertion loss and X talk\Fibre Array 2\Yellow array\170126\Port 2"
p2 = r"C:\Users\Philip\Documents\Powerpoints\IEEE Yangzhou"
data1 = prd.load_multicsv(p1)

print(np.shape(data1), np.shape(data1))
os.chdir(p1)
f1 = p1 + r'\*.mat'
files = glob.glob(f1)
data_all = np.array([])

fig1 = plt.figure('fig1')
ax1 = fig1.add_subplot(1, 1, 1)
fig1.patch.set_facecolor(cs['mdk_dgrey'])
ax1.set_xlabel('Wavelength (nm)')
ax1.set_ylabel('Insertion Loss (dB)')

for i1, val1 in enumerate(files[0:]):
    print('file name = ', val1)
    data = io.loadmat(val1)
    if int(data['fibre']) > 4:
        fibre = str(int(data['fibre']))

    else:
        fibre = str(int(data['fibre']))

    label = 'Port ' + fibre
    fibre_c = 'fibre9d_' + fibre
    print(fibre)
    print(np.transpose(np.shape(data['Ps'])))
    plt.plot(np.transpose(data['lambdas']), np.transpose(data['Ps']), '-',
             lw=1, label=label, c=cs[fibre_c])

leg1 = plt.legend(prop={'size': 6})
leg1.get_frame().set_alpha(0.0)
for text in leg1.get_texts():
    text.set_color("black")

##############################################################################
# Plot some figures
##############################################################################
os.chdir(p2)

plt.show()

prd.PPT_save_2d(fig1, ax1, 'Port2')
