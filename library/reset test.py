import numpy as np
Map_p = 'Map.txt'
Hc1_p = 'Holocs1.txt'
Hc2_p = 'Holocs2.txt'
Hd_p = 'Holod'

f0 = open('i0.txt', 'w')
f0.write(str(0))
f0.close()

f1 = open('i1.txt', 'w')
f1.write(str(1))
f1.close()

f2 = open('Map.txt','w')
f2.write(str())
f2.close()

Map = []
Hc1 = []
Hc2 = []
Hd = []
np.savetxt(Hd_p, Hd, fmt='%f', delimiter=',')
np.savetxt(Hc2_p, Hc2, fmt='%f', delimiter=',')
np.savetxt(Hc1_p, Hc1, fmt='%f', delimiter=',')