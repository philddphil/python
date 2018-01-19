#  Fitting_beam_positions.py
#
#  Code to analyse input/output array alignment
#
#  ROADMap Systems Ltd
#
#  Brian Robertson
#
#  20/November/2017
#
#  Version 1 - Basic code. (20/11/2017)


from numpy import*
from pylab import *
from math import*
from scipy.optimize import minimize

# Define functions

# Calculate theoretical beam positions


def calculate(scale, angle, ex, ey):

    error = 0

    for ii in range(3):
        ii2 = ii - 1
        for jj in range(3):
            jj2 = jj - 1

            posx = xm[ii, jj]
            posy = ym[ii, jj]

            posx_ideal = ii2 * scale * \
                np.cos(angle) + jj2 * scale * np.sin(angle)
            posy_ideal = ii2 * scale * \
                np.sin(angle) - jj2 * scale * np.cos(angle)

            x_theory[ii, jj] = posx_ideal + ex
            y_theory[ii, jj] = posy_ideal + ey

            error_new = (posx - posx_ideal - ex)**2 + \
                (posy - posy_ideal - ey)**2
            error = error + error_new

    return x_theory, y_theory, error

# Minimization function


def fun(x00):

    scale = x00[0]
    angle = x00[1]
    ex = x00[2]
    ey = x00[3]

    error = 0

    for ii in range(3):
        ii2 = ii - 1
        for jj in range(3):
            jj2 = jj - 1

            posx = xm[ii, jj]
            posy = ym[ii, jj]

            posx_ideal = ii2 * scale * \
                np.cos(angle) + jj2 * scale * np.sin(angle)
            posy_ideal = ii2 * scale * \
                np.sin(angle) - jj2 * scale * np.cos(angle)

            x_theory[ii, jj] = posx_ideal + ex
            y_theory[ii, jj] = posy_ideal + ey

            error_new = (posx - posx_ideal - ex)**2 + \
                (posy - posy_ideal - ey)**2
            error = error + error_new

    return error

# Main code

# Experimental data
x = np.array([1.407, 1.436, 1.459, 1.739, 1.784, 1.798, 2.076, 2.111, 2.103])
y = np.array([3.211, 2.820, 2.503, 3.223, 2.860, 2.513, 3.213, 2.858, 2.530])

x0 = np.array([1.183, 1.220, 1.224, 1.509, 1.561, 1.575, 1.837, 1.866, 1.873])
y0 = np.array([3.523, 3.137, 2.824, 3.531, 3.181, 2.839, 3.515, 3.168, 2.852])

x1 = np.array([0.793, 0.8309, 0.8565, 1.334,
               1.394, 1.410, 1.883, 1.934, 1.926])
y1 = np.array([3.999, 3.373, 2.836, 4.017, 3.427, 2.854, 3.989, 3.414, 2.872])


xo = x[4]
yo = y[4]

# Experimental data in array form
xm = np.array([[1.407, 1.436, 1.459], [1.739, 1.784, 1.798],
               [2.076, 2.111, 2.103]]) - xo
ym = np.array([[3.211, 2.820, 2.503], [3.223, 2.860, 2.513],
               [3.213, 2.858, 2.530]]) - yo

# Define theoretical position matrix
x_theory = np.zeros([3, 3])
y_theory = np.zeros([3, 3])

# Initial parameters (best guess)
scale = 0.35
angle = 2 * pi / 180
ex = 0
ey = 0

# Set up parameters for minimization function
x00 = [scale, angle, ex, ey]
# Optimization using function 'fun'
res = minimize(fun, x00, method='Powell')
# Optimized parameters
print(res.x)

scale = res.x[0]
angle = res.x[1]
ex = res.x[2]
ey = res.x[3]

print('Scale =', scale)
print('Angle =', angle * 180 / pi)
print('ex =', ex)
print('ey =', ey)

# Calculate theoretical positions
x_theory, y_theory, error = calculate(scale, angle, ex, ey)
print(error)

# Plot data
plot(xm, ym, 'o')
plot(x_theory, y_theory, '+')
xlabel('x-axis')
ylabel('y-axis')
show()
