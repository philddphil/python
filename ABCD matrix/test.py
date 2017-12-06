##############################################################################
# Import some libraries
##############################################################################
import sys
import numpy as np
import matplotlib.pyplot as plt


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
X = np.array([1.460, 2.500, 3.548, 4.717])
Y = np.array([3.425, 3.370, 3.395, 3.450])


fig1 = plt.figure('fig1')
ax1 = fig1.add_subplot(1, 1, 1)
fig1.patch.set_facecolor(cs['mdk_dgrey'])
ax1.set_xlabel('x axis')
ax1.set_ylabel('y axis')
plt.plot(X, Y, 'o')

plt.show()

prd.PPT_save_2d(fig1, ax1, '2D array fit 1')
