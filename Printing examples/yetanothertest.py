import numpy as np
import random
p = r"C:\Users\User\Documents\Phils LabVIEW\Data\Python loops\Anneal Hol params Last.txt"

V = np.genfromtxt(p, delimiter=',')
print(V)

np.set_printoptions(suppress=True)
print(np.round(V,3))
for i in V:
    print(np.round(i, 4))
