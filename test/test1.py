import matplotlib.pyplot as plt
import numpy as np
import scipy as sc

X = np.linspace(0, 1, 100)

Y = np.sin(X * np.pi)
plt.plot(X, Y)
plt.show()
ans1 = np.max(Y)
ans2 = np.min(Y)
print(ans1, ans2)

