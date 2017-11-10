##############################################################################
# Import some libraries
##############################################################################
import numpy as np
import matplotlib.pyplot as plt
import random

##############################################################################
# Import some extra special libraries from my own repo
##############################################################################
# sys.path.insert(0, r'C:\Users\Philip\Documents\Python\Local Repo\library')
import useful_defs_prd as prd
cs = prd.palette()

##############################################################################
# Do some stuff
##############################################################################

Map_p = 'Map.txt'
Hc1_p = 'Holocs1.txt'
Hc2_p = 'Holocs2.txt'
Hd_p = 'Holod'

Map = np.genfromtxt(Map_p, dtype='int', delimiter=',')
Hc1 = np.genfromtxt(Hc1_p, delimiter=',')
Hc2 = np.genfromtxt(Hc2_p, delimiter=',')
Hd = np.genfromtxt(Hd_p, delimiter=',')

Map = np.atleast_1d(Map)
Hc1 = np.atleast_1d(Hc1)
Hc2 = np.atleast_1d(Hc2)
Hd = np.atleast_1d(Hd)

i0 = np.genfromtxt('i0.txt', dtype='int')
i1 = np.genfromtxt('i1.txt', dtype='int')

hd = 1 / 2**(i1)

print('i1 = ', i1)
print('i0 = ', i0)
start = 0.5
shift = 0


if i1 == 1:
    start = 0.5
    print('Start shift = ', 0)
else:
    for j1 in range(i1 - 1):
        print(Map[j1], ' ', 1 / (2**(j1 + 2)), -
              (1 / (2**(j1 + 2))) * (-1)**(Map[j1]))
        shift = -(1 / (2**(j1 + 2))) * (-1)**(Map[j1]) + shift
    start = 0.5 + shift
    print('Start shift = ', shift)

print('Start = ', start)
if i0 == 0:
    Hc1 = np.append(Hc1, start - hd / 2)
    Hd = np.append(Hd, hd)

    np.savetxt(Hc1_p, Hc1, fmt='%.4f', delimiter=',')
    np.savetxt(Hd_p, Hd, fmt='%.4f', delimiter=',')

    f0 = open('i0.txt', 'w')
    f0.write(str(1))
    f0.close()

elif i0 == 1:
    mp = random.randint(0, 1)
    Map = np.append(Map, mp)
    Hc2 = np.append(Hc2, start + hd / 2)
    np.savetxt(Map_p, Map, fmt='%d', delimiter=',')
    np.savetxt(Hc2_p, Hc2, fmt='%.4f', delimiter=',')
    f0 = open('i0.txt', 'w')
    f0.write(str(0))
    f0.close()
    f1 = open('i1.txt', 'w')
    f1.write(str(i1 + 1))
    f1.close()
    print('Map = ', Map, type(Map))

    fig1 = plt.figure('fig1')
    ax1 = fig1.add_subplot(1, 1, 1)
    fig1.patch.set_facecolor(cs['mdk_dgrey'])
    ax1.set_xlabel('x axis')
    ax1.set_ylabel('y axis')

    plt.plot(Hc1 - Hd / 2, '-', c='xkcd:pale red')
    plt.plot(Hc1 + Hd / 2, '-', c='xkcd:pale red')
    plt.plot(Hc2 - Hd / 2, ':', c='xkcd:light blue')
    plt.plot(Hc2 + Hd / 2, '-', c='xkcd:light blue')

    for j2, val2 in enumerate(Map):

        if val2 == 1:
            c1 = 'red'
            c2 = 'green'
        else:
            c2 = 'red'
            c1 = 'green'

        plt.plot(j2, Hc2[j2], 'o', c=c2)
        plt.plot(j2, Hc1[j2], 'o', c=c1)

    plt.show()


print('Hc1:', Hc1)
print('Hc2:', Hc2)
print('Hd:', Hd)


# elif i0 == 1:
# 	hc = 1/2 + 1 / (2 ** (i0 + 1))

##############################################################################
# Plot some figures
##############################################################################
