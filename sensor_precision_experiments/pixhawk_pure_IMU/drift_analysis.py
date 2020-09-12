import of_library as of
import numpy as np
import matplotlib.pyplot as plt


#This script calculates the perceived motion drift derived from the Imu calculations
#These Values can later be used to compare a IMU based solution to the Optical Integration.
def integrate(a,t,k):
    vel=a[:k]*t[:k]
    return sum(vel)

    
data=of.read_yaml_imu('imuData.yaml')

time=[]
x_data=[]
y_data=[]
z_data=[]

for entries  in data:
    time.append(entries[0])
    x_data.append(entries[5].x)
    y_data.append(entries[5].y)
    z_data.append(entries[5].z)

t_diff = np.array([y-x for x,y in zip(time,time[1:])])


print('the standart deviation of x,y,z is :', np.std(x_data),np.std(y_data),np.std(z_data), 'in m/s')
print('the sample size was:',len(x_data))
print(' the median Imu frequency was',1/np.mean(t_diff),'with a standart deviation of:',1/np.std(t_diff))


#velocity calculation
#--------------------------------------------
x_vel=np.zeros(len(data)-1)
y_vel=np.zeros(len(data)-1)
z_vel=np.zeros(len(data)-1)

for i in range(len(data)-1):
    x_vel[i]=integrate(x_data,t_diff,i+1)
    y_vel[i]=integrate(y_data,t_diff,i+1)
    z_vel[i]=integrate(z_data,t_diff,i+1)


plt.scatter(np.array(time[1:])-time[0],x_vel,label='x-Richtung')
plt.scatter(np.array(time[1:])-time[0],y_vel,label='y-Richtung')
plt.scatter(np.array(time[1:])-time[0],z_vel,label='z-Richtung')
plt.legend(prop={'size':20})
plt.xlabel('Zeit [s]',fontsize=24)
plt.ylabel('Geschwindigkeit [m/s]',fontsize=24)
plt.tick_params(labelsize=20)
plt.show()
#plt.savefig('velocity_drift.png')

'''
#position calculation
#----------------------------------------
x_pos=np.zeros(len(x_vel)-1)
y_pos=np.zeros(len(x_vel)-1)
z_pos=np.zeros(len(x_vel)-1)

for i in range(len(x_vel)-1):
    x_pos[i]=integrate(x_vel,t_diff,i+1)
    y_pos[i]=integrate(y_vel,t_diff,i+1)
    z_pos[i]=integrate(z_vel,t_diff,i+1)

plt.plot(time[2:],x_pos,label='x axis')
plt.plot(time[2:],y_pos,label='y axis')
plt.plot(time[2:],z_pos,label='z axis')
plt.legend()
plt.title('position drift')
plt.xlabel('time in seconds')
plt.ylabel('Position in m')
plt.savefig('position_drift.png')
'''
