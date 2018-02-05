import sys
import numpy as np
import matplotlib.pyplot as plt
import os
sys.path.insert(0, r'C:\Users\Philip\Documents\Python\Local Repo\library')
np.set_printoptions(suppress=True)
import useful_defs_prd as prd
cs = prd.palette()


f2 = r"C:\Users\Philip\Desktop"

res = 100
IL = np.linspace(-10, -4, res)
XT = np.linspace(-40, -15, res)

M1 = np.zeros((res, res))
M2 = np.zeros((res, res))

for i1, val1 in enumerate(IL):
    for i2, val2 in enumerate(XT):
        M1[i1, i2] = 1 - (val1 + val2)
        M2[i1, i2] = 1 - (5 * val1 + val2)


fig1 = plt.figure('fig1', figsize=(5, 4))
ax1 = fig1.add_subplot(1, 1, 1)
fig1.patch.set_facecolor(cs['mdk_dgrey'])
ax1.set_xlabel('Cross talk XT (dB)')
# ax1.set_ylabel('y axis - phase/π')
ax1.set_ylabel('Insertion loss IL (dB)')


plt.imshow(M2, extent=prd.extents(XT) + prd.extents(IL))
cbar = plt.colorbar()
plt.contour(XT, IL, np.flipud(M2), 5, colors='white')

cbar.set_label('Merit function MF (dB)', rotation=270,
               color='xkcd:charcoal grey')
cbar.ax.get_yaxis().labelpad = 15
ax1.set_aspect(4)

plt.show()
os.chdir(f2)
prd.PPT_save_2d_im(fig1, ax1, cbar, 'MF map.png')
