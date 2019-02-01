'''
velcoity drift of a standing uav using optical stabilisation
'''

import numpy as np
import matplotlib.pyplot as plt

x, y, z,t=np.loadtxt('velocity_no_cam.txt',unpack=True)

plt.plot(t,x)
plt.plot(t,y)
plt.plot(t,z)
plt.show()
