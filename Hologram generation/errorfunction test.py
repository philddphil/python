
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf


sys.path.insert(0, r'C:\Users\Philip\Documents\Python\Local Repo\library')
np.set_printoptions(suppress=True)
import useful_defs_prd as prd
from peakdetect import peakdetect
cs = prd.palette()


w = 0.1
x = np.linspace(-1/w, 1/w, 100)
z = erf(x)

fig1 = plt.figure('fig1', figsize=(4, 4))
ax1 = fig1.add_subplot(1, 1, 1)
fig1.patch.set_facecolor(cs['mdk_dgrey'])
ax1.set_xlabel('x axis - greylevel')
ax1.set_ylabel('y axis - phase/π')
plt.plot(x, z, 'o:')
plt.show()
