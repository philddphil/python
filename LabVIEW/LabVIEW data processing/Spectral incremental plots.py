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
p1 = r"C:\Users\Philip\Documents\Powerpoints\IEEE Yangzhou"
p2 = (r"C:\Users\Philip\Documents\Technical Stuff\Hologram optimisation"
	  r"\High frequency sin term\180228\Sin amp values")
f1 = p2 + r'\*.csv'
files = glob.glob(f1)
data_all = np.array([])
print(files)


for i1, val1 in enumerate(files[:]):
	print(i1)
	data = np.genfromtxt(val1, delimiter=',')
	print(np.shape(data))
	fig1 = plt.figure('fig1')
	ax1 = fig1.add_subplot(1, 1, 1)
	fig1.patch.set_facecolor(cs['mdk_dgrey'])
	ax1.set_xlabel('Frequency (GHz)')
	ax1.set_ylabel('Signal (dB)')
	plt.plot(1.0 * data[0, :], data[1, :], '-',
			 lw=1, c=cs['ggred'])
	ax1.set_ylim([-70, -25])
	plt.show()
	os.chdir(p2)
	prd.PPT_save_2d(fig1, ax1, str(i1))

# leg1 = plt.legend()
# leg1.get_frame().set_alpha(0.0)

##############################################################################
# Plot some figures
##############################################################################
