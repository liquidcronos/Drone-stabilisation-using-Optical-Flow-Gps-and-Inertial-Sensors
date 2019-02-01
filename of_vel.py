#!/usr/bin/env python
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Quaternion
import rospy
import numpy as np
import scipy.linalg
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
def solve_lgs(x,u,d,n,omega):
    A=np.empty((0,3))
    B=np.empty(0)
    #better would be to do it in parallel
    for i in range(len(x)):
        x_hat=np.array([[0,-1,x[i,1]],[1,0,-x[i,0]],[-x[i,1],x[i,0],0]]) 
        b_i = np.dot(x_hat,np.array([u[i,0],u[i,1],0])+np.dot(x_hat,omega))/np.dot(n,np.append(x[i],1))  #append 3rd dim for calculation (faster method ?)
        A=np.append(A,x_hat,axis=0)
        B=np.append(B,b_i)
    try:
        return np.linalg.lstsq(A/d,B)   #v,R,rank,s
    except:
        return np.zeros(3), 10000 * np.ones(3*len(x))

def feasible(x,v,omega,T,u,d,n):
    v_cross       = [np.cross(np.append(x[i,:],1),v+np.cross(omega,T)) for i in range(len(x))]
    u_cross       = [np.cross(np.append(x[i,:],0),np.append(u[i,:],0))  for i in range(len(x))]
    parallelity   = [np.dot(v_cr,u_cr)/(np.linalg.norm(v_cr)*np.linalg.norm(u_cr)) if np.linalg.norm(v_cr)*np.linalg.norm(u_cr) != 0 
                     else 1 for v_cr,u_cr in zip(v_cross,u_cross)]
    distance      = [np.dot(n,np.append(x[i,:],1))*np.linalg.norm(v_cr)/np.linalg.norm(u_cr)/d if np.linalg.norm(u_cr) != 0
                     else 0 for i,v_cr,u_cr in zip(np.linspace(0,len(v_cr)),v_cr,u_cr)]

class optical_fusion:


    def call_imu(self,data):
        self.ang      = np.array([data.angular_velocity.x,data.angular_velocity.y,data.angular_velocity.z])
        self.ang_err  = np.array([data.angular_velocity_covariance[0],data.angular_velocity_covariance[4],data.angular_velocity_covariance[8]])

        q                     = data.orientation
        R             = np.array([[1.0-2*(q.y**2+q.z**2),2*(q.x*q.y-q.w*q.z),2*(q.w*q.y+q.x*q.z)],
                                  [2*(q.x*q.y+q.w*q.z),1.0-2*(q.x**2+q.z**2),2*(q.y*q.z-q.w*q.x)],
                                  [2*(q.x*q.z-q.w*q.y),2*(q.w*q.x+q.y*q.z),1.0-2*(q.x**2+q.y**2)]])
        self.rotation = R
        self.normal   = np.dot(R,np.array([0,0,1]))     #if camera was tilted: !=1
        if self.first_imu_:
            self.old_time   = float(data.header.stamp.nsecs)/10**9
            self.time_zero  = data.header.stamp.secs
            self.first_imu_ = False
        else:

            current_time    = float(data.header.stamp.secs-self.time_zero)+float(data.header.stamp.nsecs)/10**9
            elapsed_time    = current_time-self.old_time

            if not self.got_vel_ :
                self.vel        = self.vel+np.dot(R,np.array([data.linear_acceleration.x,data.linear_acceleration.y,data.linear_acceleration.z])-
                                  9.81*self.normal)*elapsed_time
                
                print_test=np.append(self.vel,current_time)
                #print(' '.join(map(str,print_test)))

            self.old_time   = float(current_time)

        self.got_ang_vel_   = True

    #returns features and flow in px where (0,0) is the top right
    def call_optical(self,image_raw):

        #parameters--------------------------------------------------
        rate     = rospy.Rate(20)    #updaterate in Hz
        min_feat = 30               #minimum number of features 
        max_feat = 100              #maximum number of features

        #Parameters for corner Detection
        feature_params = dict( qualityLevel = 0.3,
                               minDistance = 30,  #changed from 7
                               blockSize = 14 )  #changed from 7

        # Parameters for lucas kanade optical flow
        lk_params      = dict( winSize  = (30,30),   #changed from (15,15)
                               maxLevel = 3,       #changed from 1
                               criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)) #changed from 10,0.03
        #------------------------------------------------------------------------------
        if self.got_picture_==False:
            bridge=CvBridge()
            image=bridge.compressed_imgmsg_to_cv2(image_raw,'bgr8') #TODO convert straight to b/w
            image_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

            old_pos=self.feat.reshape((len(self.feat),1,2))
            old_pos_err=self.feat_err
            if  self.first == True:
                first_feat =cv2.goodFeaturesToTrack(image_gray,mask=None,maxCorners=max_feat,**feature_params)

                #Test data 
                #first_feat    = np.array([[401,300],[399,300],[400,301],[400,299]])
                #self.feat_err = 0.01*np.ones((4,2))

                self.feat=first_feat.reshape((len(first_feat),2))
                self.feat_err=np.zeros(len(first_feat))

            else:
                if len(old_pos) <= min_feat: 

                    ft_mask=np.ones_like(image_gray)
                    for i in range(len(old_pos)):
                        cv2.circle(ft_mask,(old_pos[i,0,0],old_pos[i,0,1]),30,0,cv2.FILLED)
                    cv2.imwrite("cubism.jpg",255*ft_mask)

                    new_features = cv2.goodFeaturesToTrack(self.old_pic,mask=ft_mask,maxCorners=max_feat-len(old_pos),**feature_params)

                    if len(new_features) >=1:
                        old_pos      = np.append(old_pos,new_features,axis=0)
                        old_pos_err  = np.append(old_pos_err,np.zeros(len(new_features)))

                #''' 
                new_pos,status,new_pos_err = cv2.calcOpticalFlowPyrLK(self.old_pic,image_gray,old_pos,None,**lk_params)
                self.feat                  = new_pos[status==1].reshape((len(new_pos[status==1]),2))
                self.feat_err              = new_pos_err[status==1]
                self.flow                  = new_pos[status==1]-old_pos[status==1]
                old_pos_err                = old_pos_err.reshape(len(old_pos_err),1)
                self.flow_err              = np.sqrt(new_pos_err[status==1]**2+old_pos_err[status==1]**2)
                #'''



                #self.flow=np.array([[1.03,0],[-1.02,0],[0,0.995],[0,-1.02]])
                #self.flow_err=0.01*np.ones((4,2))
                #self.flow=np.ones((4,2))
                #self.flow_err=0.01*np.ones((4,2))
                #self.flow=self.flow+np.ones((4,2))

                self.init         = False 
                self.got_picture_ = True 

            self.old_pic=image_gray
            self.first=False
            rate.sleep()

 
    def __init__(self):
        #TODO init values in numpy, not true data type
        scaling          = 0.01    #needs to be calibrated
        self.vel         = np.array([0.1,0.1,0.1])  #trans vel.
        self.vel_err     = np.array([0.1,0.1,0.1])  #trans vel.

        self.feat        = np.ones((1,2))  #array of features
        self.feat_err    = np.ones((1,1)) 

        self.flow        = np.zeros((1,1,2))  #array of flows
        self.flow_err    = np.zeros((1,1,2))  

        self.ang         = np.array([0,0,0])
        self.ang_err     = np.zeros((3,3))

        self.d           = 0.75     #distance in m
        self.d_err       = 0

        self.rotation    = np.array([[1,0,0],[0,1,0],[0,0,1]])

        self.old_pic     = np.zeros((480,640)) # last picture for calculating OF
        self.old_time    = 0      #dummy value for first time
        self.time_zero   = 0

        self.offset      = np.array([0,0,0.1])

        self.normal      = np.array([0,0,1])
        self.normal_err  = np.array([0,0,1])


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
            if self.got_picture_ and not self.init:
                #zero picture coordinates before solving lgs
                translation=of.pix_trans((1280,960))
                x=copy.deepcopy(self.feat)
                x=x.astype(float)
                x[:,0]=(x[:,0]-translation[0])*scaling
                x[:,1]=(x[:,1]-translation[1])*scaling

                u=self.flow.reshape(len(self.flow),2)*scaling #copy flow
                
                feasibility,dummy_d = of.r_tilde(x,u,self.normal,self.vel,self.d) #TODO distance calc. faulty
                #print(feasibility)
                feasibility= -1*np.ones(len(x)) #hardcoded to one to test while system stationary
                T         = 0.0  #feasibility Treshold
                x         = x[feasibility <= T]
                u         = u[feasibility <= T]
                dummy_d   = dummy_d[feasibility<=T]
                self.feat = self.feat[feasibility <= T]

                feasible(self.feat,self.vel,self.ang,self.offset,u,self.d,self.normal)
                if len(x) >= 3:
                    #statistical analysis and d calculation
                    d_sorted=np.sort(dummy_d)
                    d_diff=[j-i for i, j in zip(d_sorted[:-1],d_sorted[1:])]
                    #splits=d_diff>=d_exp_err



                    if len(x) >=3:
                        v_obs,R,rank,s = solve_lgs(x,u,self.d,self.normal,self.ang)
                        v_uav          = np.dot(self.rotation,v_obs-np.dot(np.array([[0,-self.ang[2],self.ang[1]],[self.ang[2],0,-self.ang[0]],[-self.ang[1],self.ang[0],0]]),self.offset))
                        print('    '.join(map(str,v_obs)))
                        got_vel_       = True
                        self.vel       = v_uav
                        got_vel_       = False
                        #print("Residuum:",np.sqrt(R/len(x)))

                self.got_picture_=False


if __name__=='__main__':
    print('++')
    rospy.init_node('velocity_calc')
    print('--')
    try:
        node=optical_fusion()
    except rospy.ROSInterruptException: pass

