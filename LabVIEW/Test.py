##############################################################################
# Import some libraries
##############################################################################

import re
import numpy as np

##############################################################################
# Do some stuff
##############################################################################


s1 = re.findall(r'[+-]?\d+[\.]?\d*', s0)
s2 = re.findall(r'\s(\D*)', s0)

print(type(s1))
print(type(s2))

print(len(s1))
print(len(s2))

for i1 in range(len(s1)):
    print(s2[i1], ' = ', s1[i1])

f1 = [float(i1) for i1 in s1]
print(2 * f1[0])
