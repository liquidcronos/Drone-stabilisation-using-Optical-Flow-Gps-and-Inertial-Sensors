#!/usr/bin/env python
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Quaternion
import rospy
import numpy as np
import cv2
import copy
from cv_bridge import CvBridge, CvBridgeError
import of_library as of
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Imu
#x: n,2 array of feature pos
#u: n,2 array of feature flows
#d: proposed distance to plain
#TODO move to of_library
def solve_lgs(x,ui,d,n):
    A=np.empty((0,3))
    B=np.empty(0)
    u=zip(ui[0],ui[1])
    #better would be to do it in parallel
    for i in range(len(x)):
        x_hat=np.array([[0,-1,x[i,1]],[1,0,-x[i,0]],[-x[i,1],x[i,0],0]])
        b_i = np.dot(x_hat,np.append(u[i],0))/np.dot(n,np.append(x[i],1))  #append 3rd dim for calculation (faster method ?)
        A=np.append(A,x_hat,axis=0)
        B=np.append(B,b_i)

    #if linalg system is non sovlable
    try:
        return np.linalg.lstsq(A/d,B)   #v,R,rank,s
    except:
        return np.zeros(3), 10000 * np.ones(3*len(x))



class optical_fusion:


    def call_imu(self,data):
        self.ang=data.angular_velocity
        self.orient=data.orientation 


        q=self.orient
        R=np.array([[1-2*(q.y**2+q.z**2),2*(q.x*q.y-q.w*q.z),2*(q.w*q.y+q.x*q.z)],
                    [2*(q.x*q.y+q.w*q.z),1-2*(q.x**2+q.z**2),2*(q.y*q.z-q.w*q.x)],
                    [2*(q.x*q.z-q.w*q.y),2*(q.w*q.x+q.y*q.z),1-2*(q.x**2+q.y**2)]])
        self.normal=np.dot(R,np.array([0,0,01]))
        #calculate current speed based on previous speed
        #Rotation matrix

        if self.first_imu_:
            self.old_time=float(data.header.stamp.nsecs)/10**9
            self.time_zero=data.header.stamp.secs
            self.first_imu_=False
        else:

            current_time=float(data.header.stamp.secs-self.time_zero)+float(data.header.stamp.nsecs)/10**9
            elapsed_time=current_time-self.old_time
            ##print(elapsed_time)
            #print(self.old_time)
            self.vel=self.vel+np.dot(R,np.array([data.linear_acceleration.x,data.linear_acceleration.y,data.linear_acceleration.z])-9.81*self.normal)*elapsed_time
            self.old_time=float(current_time)
        self.got_ang_vel_=True

    #camera listener. input: image from ros
    #returns features and flow in px where (0,0) is the top right
    def call_optical(self,image_raw):

        #parameters--------------------------------------------------
        min_feat= 30  #minimum number of features 
        max_feat=60   #maximum number of features

        #Parameters for corner Detection
        feature_params = dict( qualityLevel = 0.3,
                               minDistance = 10,  #changed from 7
                               blockSize = 32 )  #changed from 7

        # Parameters for lucas kanade optical flow
        lk_params = dict( winSize  = (15,15),   #changed from (15,15)
                          maxLevel = 3,       #changed from 1
                          criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.5)) #changed from 10,0.03
        #------------------------------------------------------------------------------

        bridge=CvBridge()        
        image=bridge.compressed_imgmsg_to_cv2(image_raw,'bgr8') #TODO convert straight to b/w
        image_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        old_pos=self.feat.reshape((len(self.feat),1,2))
        #generate new features if to few where found
        if len(old_pos) <= min_feat: 
            #generate old image on initialisation
            if  self.first == True:
                self.feat =cv2.goodFeaturesToTrack(image_gray,mask=None,maxCorners=max_feat,**feature_params) 
            else:
                old_pos = np.append(old_pos,
                                    cv2.goodFeaturesToTrack(self.old_pic,mask=None,maxCorners=max_feat-len(old_pos),**feature_params),axis=0)

        if  self.first== False:
            old_pol=old_pos.reshape((len(old_pos),2))
            new_pos,status,new_pos_err = cv2.calcOpticalFlowPyrLK(self.old_pic,image_gray,old_pos,None,**lk_params)
            self.feat=new_pos[status==1].reshape((len(new_pos[status==1]),2))
            #will lead to problems if length of new_pos is changed
            self.flow=new_pos[status==1]-old_pos[status==1]
            self.init         = False 

        self.old_pic=image_gray
        #confirm that a picture has been taken
        self.got_picture_ = True 
        self.first=False

 
    def __init__(self):
        #TODO init values in numpy, not true data type
        self.vel    = np.array([0.1,0.1,0.1])  #trans vel.
        self.feat   = np.ones((1,2))  #array of features

        self.flow   = np.zeros((1,1,2))  #array of flows
        self.ang    = Vector3(0,0,0)  #Vector3
        self.orient = Quaternion(0,0,0,0)
        self.d      =  1     #distance in m
        self.old_pic= np.zeros((480,640)) # last picture for calculating OFi
        self.old_time=0      #dummy value for first time
        self.normal = np.array([0,0,1])
        self.time_zero=0

        #flags
        self.init        = True
        self.first       = True
        self.first_imu_  = True
        self.got_normal_ = False
        self.got_vel_    = False
        self.got_picture_= False
        self.got_ang_vel_= False

        rospy.Subscriber('/mavros/imu/data', Imu, self.call_imu)
        rospy.Subscriber('/camerav2_1280x960/image_raw/compressed', CompressedImage,self.call_optical)
        while not rospy.is_shutdown():
            #implement flag handling and publishing at certain times
            #solve lgs here -> Time handling ??  
            if self.got_picture_ and not self.init:
                #zero picture coordinates before solving lgs
                translation=of.pix_trans((480,640))
                x=copy.deepcopy(self.feat)
                x[:,0]=x[:,0]-translation[0]
                x[:,1]=x[:,1]-translation[1]

                u=self.flow.reshape(len(self.flow),2) #copy flow
                #calculate feasible points
                #!!Carefull use velocity at time of picture save v_vel in other case !!!

                normal = self.normal
                feasibility,dummy_d = of.r_tilde(x,u,normal,self.vel) #TODO distance calc. faulty
                #print(feasibility)
                #feasibility= -1*np.ones(len(x)) #hardcoded to one to test while system stationary
                T=-0.5
                x=x[feasibility <= T]
                u=u[feasibility <= T]
                
                #Problem feasibility needs to be corrected, but image subscriber could change self.feat at the same time...
                self.feat=self.feat[feasibility <= T]
                
                if len(x) >= 3:
                    u=np.array([u[:,0] - x[:,0]*x[:,1]*self.ang.x+(1+x[:,0]**2)*self.ang.y-x[:,1]*self.ang.z,
                                u[:,1] +(1+x[:,1]**2)*self.ang.x+x[:,0]*x[:,1]*self.ang.y+x[:,0]*self.ang.z])

                    v_obs,R,rank,s= solve_lgs(x,u,self.d,normal)

                    #TODO implement kalman filter to combine IMU and optical measurment
                    print("Observed Speed:",v_obs)
                    print("IMU Speed", self.vel)
                    #print(R/len(x)) 
                self.got_picture_=False


if __name__=='__main__':
    print('++')
    rospy.init_node('velocity_calc')
    print('--')
    try:
        node=optical_fusion()
    except rospy.ROSInterruptException: pass

