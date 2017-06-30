import numpy as np
import matplotlib.pyplot as plt

path = r"C:\Users\Philip\Desktop\LiveMode.csv"

my_data = np.genfromtxt(path, delimiter=',')

plt.imshow(my_data)

x = plt.ginput(-1, show_clicks=True, mouse_add=1, mouse_pop=3, mouse_stop=2)

plt.show()

print(x)
