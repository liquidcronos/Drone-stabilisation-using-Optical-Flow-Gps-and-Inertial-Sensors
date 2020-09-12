import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import yaml
import rospy


#1ster parameter = stelle der Messung
streamCam = file('imuData.yaml' , 'r')
dataCam = yaml.load(streamCam)

first_meas=dataCam[0]
#get acceleration
print(first_meas.linear_acceleration)
print(first_meas.angular_velocity)

#all data:
for data in dataCam:
    print(data)


