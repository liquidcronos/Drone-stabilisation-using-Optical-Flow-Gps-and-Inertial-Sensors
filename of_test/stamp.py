import of_library as of
import numpy as np
import matplotlib.pyplot as plt

daten=of.read_yaml_imu('BeispielDatenImuCam22-10-18/imuData.yaml')

timestamb=[]
for data in daten:
    timestamb.append(data[0])

#plt.hist(timestamb,bins=100)
plt.scatter(np.arange(0,len(timestamb)),timestamb)
plt.show()
