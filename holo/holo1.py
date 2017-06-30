import matplotlib.pyplot as plt
import numpy as np
import copy

min_ph = 0.0
max_ph = 3.0
LCOSy = 100
LCOSx = 20
θ = np.arctan((1 * (max_ph - min_ph)) / LCOSx)
ϕ = 6*np.pi / 4


x = np.arange(LCOSx)
y = np.arange(LCOSy)

[X, Y] = np.meshgrid(x, y)

Z = X * np.sin(θ) * np.cos(ϕ) + Y * np.sin(θ) * np.sin(ϕ)
Z_mod = np.mod(Z, max_ph - min_ph) + min_ph
cmap = plt.get_cmap('binary')

im = plt.imshow(Z_mod, cmap)
plt.imshow
plt.savefig('holo1.png', transparent=True)

plt.show()



