import numpy as np
import rospy
import cv2
import scipy.linalg
import matplotlib.pyplot as plt
import yaml


from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Imu
from sensor_msgs.msg import Range
from geometry_msgs.msg import  Vector3
from geometry_msgs.msg import Quaternion



def solve_lgs(x,u,d,n,omega,t):
    A=np.empty((0,3))
    B=np.empty(0)
   #better would be to do it in parallel
    for i in range(len(x)):
        x_hat=np.array([[0,-1.0,x[i,1]],[1.0,0,-x[i,0]],[-x[i,1],x[i,0],0]])
        b_i = np.dot(x_hat,np.array([u[i,0],u[i,1],0])+1.00*np.dot(x_hat,omega))/np.dot(n,np.append(x[i],1))  #append 3rd dim for calculation (faster method ?)
        A=np.append(A,x_hat,axis=0)
        B=np.append(B,b_i)
    try:
        v,R,rank,s = np.linalg.lstsq(A,B*d)   #v,R,rank,s
        return v-np.cross(omega,t),R
    except:
        return np.zeros(3), 10000 * np.ones(3*len(x))


#finds closest element to later match imu to time
#parameters--------------------------------------------------
#rate     = rospy.Rate(20)    #updaterate in Hz
min_feat = 5               #minimum number of features
max_feat = 20              #maximum number of features

#Parameters for corner Detection
feature_params = dict( qualityLevel = 0.7,
                   minDistance = 10,  #changed from 7
                   blockSize = 7 )  #changed from 7

# Parameters for lucas kanade optical flow
lk_params      = dict( winSize  = (15,15),   #changed from (15,15)
                   maxLevel = 3,       #changed from 1
                   criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)) #changed from 10,0.03

Translation = np.array([0,0,1])
#------------------------------------------------------------------------------

camData=file('camData.yaml','r')
imuData=file('imuData.yaml','r')
hgtData=file('hgtData.yaml','r')
bridge=CvBridge()
test=yaml.load(camData)
imu=yaml.load(imuData)
height=yaml.load(hgtData)
np.set_printoptions(threshold=np.nan)
print(bridge.compressed_imgmsg_to_cv2(test[0]).shape)


first_image=bridge.compressed_imgmsg_to_cv2(test[479])
old_image=cv2.cvtColor(first_image,cv2.COLOR_BGR2GRAY)
old_pos =cv2.goodFeaturesToTrack(old_image,mask=None,maxCorners=max_feat,**feature_params)

imu_values=np.zeros(len(imu))
for i in range (len(imu)):
    imu_values[i]=float(imu[i].header.stamp.secs-test[0].header.stamp.secs)+float(imu[i].header.stamp.nsecs)/10**9


hgt_values=np.zeros(len(height))
for i in range (len(height)):
    hgt_values[i]=float(height[i].header.stamp.secs-test[0].header.stamp.secs)+float(height[i].header.stamp.nsecs)/10**9

for i in range(480,790):
    current_time=float(test[i].header.stamp.secs-test[0].header.stamp.secs)+float(test[i].header.stamp.nsecs)/10**9
    imu_index=np.argmin(np.abs(imu_values-current_time))
    hgt_index=np.argmin(np.abs(hgt_values-current_time))
    print(hgt_index)
    dist=height[hgt_index].range
    print(dist)
    current_image=bridge.compressed_imgmsg_to_cv2(test[i])
    image_gray=cv2.cvtColor(current_image,cv2.COLOR_BGR2GRAY)
   
    # calculate orientation:
    q      = imu[imu_index].orientation
    R      = np.array([[1.0-2*(q.y**2+q.z**2),2*(q.x*q.y-q.w*q.z),2*(q.w*q.y+q.x*q.z)],
                   [2*(q.x*q.y+q.w*q.z),1.0-2*(q.x**2+q.z**2),2*(q.y*q.z-q.w*q.x)],
                  [2*(q.x*q.z-q.w*q.y),2*(q.w*q.x+q.y*q.z),1.0-2*(q.x**2+q.y**2)]])
    normal = np.dot(R,np.array([0,0,1]))
    
    #save angular velocity to work with numpy
    omega=np.array([imu[imu_index].angular_velocity.x,imu[imu_index].angular_velocity.y,imu[imu_index].angular_velocity.z])
    print(omega)
    if len(old_pos) >= 1:
        new_pos,status,new_pos_err=cv2.calcOpticalFlowPyrLK(old_image,image_gray,old_pos,None,**lk_params)
        new_pos=new_pos[status==1].reshape((len(new_pos[status==1]),1,2))
    else:
        visualisation=current_image
    for i in range(len(new_pos)):
        visualisation=cv2.circle(current_image,(new_pos[i,0,0],new_pos[i,0,1]),10,50,cv2.FILLED)

    if len(new_pos) <= min_feat:
        new_feat=cv2.goodFeaturesToTrack(image_gray,mask=None,maxCorners=max_feat,**feature_params)
        new_pos=np.append(new_pos,new_feat,axis=0)
        if len(new_feat) >=2:
            for i in range(len(new_feat)):
                visualisation=cv2.circle(visualisation,(new_feat[i,0,0],new_feat[i,0,1]),10,100,cv2.FILLED) 


    vel_obs, r =solve_lgs(new_pos,new_pos-old_pos,dist,normal,omega,Translation)
    print(vel_obs)
    #visualisation
    cv2.imshow("frame",visualisation)
    cv2.waitKey(100)

    old_image=image_gray
    old_pos=new_pos

#cv.imshow("frame",video[800])

