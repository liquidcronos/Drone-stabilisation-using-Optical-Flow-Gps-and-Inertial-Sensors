import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

def function(x):
    return 1/(2*x**2+1)*np.exp(-x**2/(2*0.056**2))/np.sqrt(2*np.pi*0.056**2)

print(integrate.quad(function,-0.3,0.3))
plt.plot(np.linspace(-10,10,1000),function(np.linspace(-10,10,1000)))
plt.show()

