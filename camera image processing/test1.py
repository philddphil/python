import matplotlib.pyplot as plt
import random
import os
import glob
import numpy as np
import scipy.io as io
import pylab as pl
import scipy.optimize as opt
from scipy.interpolate import interp1d
import copy


def New_mod(t, x_lower, x_upper):
    t1 = t.ravel()
    print(np.shape(t1))
    print(len(t1))
    t2 = copy.copy(t1)
    for i1 in range(len(t1)):
        print(i1)
        if t1[i1] <= x_upper:
            t2[i1] = t1[i1]
            print('n')
        else:
            t2[i1] = t1[i1] % x_upper
            print('m')

    t2 = t2.reshape(*np.shape(t))
    return t2


ϕ_min = 0
ϕ_max = 10


x1 = np.arange(48)
x2 = np.arange(48)
y1 = np.zeros(len(x1))
y2 = np.zeros(len(x1))
y3 = np.zeros(len(x1))

y1 = x1 % ϕ_max
y2 = np.clip(x1, ϕ_min, ϕ_max)
y3 = New_mod(x2, ϕ_min, ϕ_max)

# plt.plot(x, y1, '.b:')
# plt.plot(x, y2, '.r:')
# plt.plot(x, y3, '.g:')

plt.plot(y1, '.:')
plt.plot(y2, '.:')
plt.plot(y3, '.:')

# print(np.max(y1), np.max(y3))
# print(np.shape(y1),np.shape(y2),np.shape(y3),np.shape(x1))

plt.show()
