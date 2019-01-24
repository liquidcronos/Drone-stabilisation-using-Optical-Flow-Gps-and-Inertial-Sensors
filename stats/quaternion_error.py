'''
since the pixhawk provided covariance doesnt seem to be working,
we provide a error estimate through the quaternion standart deviation,
while stationary.
'''

import numpy as np
import matplotlib.pyplot as plt

w,x,y,z=np.loadtxt('quaternions.txt',unpack=True)

print("Die statistischen Standartabweichungen sind:",np.std(w),np.std(x),np.std(y),np.std(z))
