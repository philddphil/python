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
p0 = r"C:\Users\Philip\Documents\Data\Hologram optimisation\OFC data\171006\Phi centre and width"
os.chdir(p0)

# f0 = os.path.join(p0, r"P0.csv")
# f1 = os.path.join(p0, r"P1.csv")
f0 = os.path.join(p0, r"Lw_lim.csv")
f1 = os.path.join(p0, r"Up_lim.csv")
f2 = os.path.join(p0, r"CT400.csv")
f3 = os.path.join(p0, r"PicoLog.csv")

D0 = np.genfromtxt(f0, delimiter=',')
D1 = np.genfromtxt(f1, delimiter=',')
D2 = np.genfromtxt(f2, delimiter=',')
D3 = np.genfromtxt(f3, delimiter=',')

y = D0[:, 0]
x = D1[0, :]

Z0 = D2
Z1 = np.abs(D3 - D2)
##############################################################################
# Plot some figures
##############################################################################

# fig0 = plt.figure('fig0')
# ax0 = Axes3D(fig0)
# fig0.patch.set_facecolor(cs['mdk_dgrey'])
# ax0.w_xaxis.set_pane_color(cs['mdk_dgrey'])
# ax0.w_yaxis.set_pane_color(cs['mdk_dgrey'])
# ax0.w_zaxis.set_pane_color(cs['mdk_dgrey'])
# ax0.set_xlabel('x axis')
# ax0.set_ylabel('y axis')
# ax0.set_zlabel('z axis')
# surf0 = ax0.plot_surface(D0, D1, D2, alpha=0.5, cmap='viridis')
# wire0 = ax0.plot_wireframe(D0, D1, D2, lw=0.5, color=cs['mdk_lgrey'])

im1 = plt.figure('im1')
ax1_1 = im1.add_subplot(1, 1, 1)
im1.patch.set_facecolor(cs['mdk_dgrey'])
ax1_1.set_xlabel('Phase range (/π)')
ax1_1.set_ylabel('Phase centre (/π)')
plt.imshow(Z0, label='Insertion Loss', extent=prd.extents(x) + prd.extents(y))
cb1 = plt.colorbar()
ax1_1.set_aspect('auto')
plt.legend()
plt.title('Insertion Loss', color=cs['mdk_dgrey'])

im2 = plt.figure('im2')
ax2_1 = im2.add_subplot(1, 1, 1)
im2.patch.set_facecolor(cs['mdk_dgrey'])
ax2_1.set_xlabel('Phase range (/π)')
ax2_1.set_ylabel('Phase centre (/π)')
plt.imshow(Z1, label='X talk', extent=prd.extents(x) + prd.extents(y))
cb2 = plt.colorbar()
ax2_1.set_aspect('auto')
plt.legend()
plt.title('X Talk', color=cs['mdk_dgrey'])

im3 = plt.figure('im3')
ax3_1 = im3.add_subplot(1, 1, 1)
im3.patch.set_facecolor(cs['mdk_dgrey'])
ax3_1.set_xlabel('Phase range (/π)')
ax3_1.set_ylabel('Phase centre (/π)')
plt.imshow(D3, label='X talk', extent=prd.extents(x) + prd.extents(y))
cb3 = plt.colorbar()
ax3_1.set_aspect('auto')
plt.legend()
plt.title('X Talk', color=cs['mdk_dgrey'])

plt.show()

prd.PPT_save_2d_im(im1, ax1_1, cb1, 'Insertion loss.png')
prd.PPT_save_2d_im(im2, ax2_1, cb2, 'X talk.png')
prd.PPT_save_2d_im(im3, ax3_1, cb3, '-1 order.png')
