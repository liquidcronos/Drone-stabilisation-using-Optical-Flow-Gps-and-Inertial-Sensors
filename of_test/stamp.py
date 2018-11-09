import of_library as of
import numpy as np
import matplotlib.pyplot as plt

daten=of.read_yaml_imu('BeispielDatenImuCam22-10-18/imuData.yaml')

#print(daten)
timestamb=[]
for data in daten:
    timestamb.append(data[0])

#print(timestamb)
plt.hist(timestamb,bins=100)
#plt.scatter(np.arange(0,len(timestamb))[0:100],timestamb[0:100])
plt.show()
