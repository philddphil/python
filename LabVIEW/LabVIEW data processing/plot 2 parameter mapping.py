##############################################################################
# Import some libraries
##############################################################################
import sys
import os
import numpy as np

import matplotlib.pyplot as plt


##############################################################################
# Import some extra special libraries from my own repo
##############################################################################
sys.path.insert(0, r'C:\Users\Philip\Documents\Python\Local Repo\library')
import useful_defs_prd as prd
cs = prd.palette()

##############################################################################
# Do some stuff
##############################################################################
p0 = (r"C:\Users\Philip\Documents\Technical Stuff\Hologram optimisation"
      r"\Phase limit variation\180205\Phase limits 1")
f1 = p0 + r"\P0.csv"
f2 = p0 + r"\P1.csv"
f3 = p0 + r"\CT400.csv"
f4 = p0 + r"\PicoLog.csv"

d0 = np.genfromtxt(f1, delimiter=',')
d1 = np.genfromtxt(f2, delimiter=',')
d2 = np.genfromtxt(f3, delimiter=',')
d3 = np.genfromtxt(f4, delimiter=',')

# ##############################################################################
# # Plot some figures
# ##############################################################################
x = d1[0, :]
y = d0[:, 0]

XT = d3 - d2

fig1 = plt.figure('fig1')
ax1 = fig1.add_subplot(1, 1, 1)
fig1.patch.set_facecolor(cs['mdk_dgrey'])
ax1.set_xlabel('P1 - sin(ϕ) offset')
ax1.set_ylabel('P0 - sin(ϕ) amplitude')
plt.imshow(XT, aspect='auto', interpolation='none',
           extent=prd.extents(x) + prd.extents(y), origin='lower')
plt.colorbar()

fig2 = plt.figure('fig2')
ax2 = fig2.add_subplot(1, 1, 1)
fig2.patch.set_facecolor(cs['mdk_dgrey'])
ax2.set_xlabel('P1 - sin(ϕ) offset')
ax2.set_ylabel('P0 - sin(ϕ) amplitude')
plt.imshow(d3, aspect='auto', interpolation='none',
           extent=prd.extents(x) + prd.extents(y), origin='lower')
plt.colorbar()

plt.show()
os.chdir(p0)
prd.PPT_save_2d(fig1, ax1, 'figure1')
