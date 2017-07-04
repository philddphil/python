##############################################################################
# Import some libraries
##############################################################################


import random
import os
import glob
import copy
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as io
import pylab as pl
import scipy.optimize as opt
import scipy.misc

from scipy.interpolate import interp1d


##############################################################################
# Define some functions
##############################################################################

##############################################################################
# Do some stuff
##############################################################################

X1 = np.array([[1, 2, 3, 4, 5, 6],[6, 5, 4, 3, 2, 1]])
thresh1 = 4
thresh2 = 3
print(X1)

X2 = copy.copy(X1)

super_threshold_indices = X1 > thresh1
sub_threshold_indices = X1 < thresh2
X2[super_threshold_indices] = 6
X2[sub_threshold_indices] = 0
print(X1)
print(X2)