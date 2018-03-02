import scipy as sp
import numpy as np 
import matplotlib.pyplot as plt

π = np.pi
Θ = np.linspace(0,2*π,100)
y = np.sin(Θ)

plt.plot(Θ,y)
plt.show()
