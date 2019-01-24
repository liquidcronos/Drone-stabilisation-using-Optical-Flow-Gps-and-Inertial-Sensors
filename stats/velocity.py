'''
velcoity drift of a standing uav using optical stabilisation
'''

import numpy as np
import matplotlib.pyplot as plt

x, y, z=np.loadtxt('velocity.txt',unpack=True)

plt.plot(x)
plt.plot(y)
plt.plot(z)
plt.show()
