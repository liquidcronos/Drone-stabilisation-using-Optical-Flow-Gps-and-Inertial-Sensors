#!/usr/bin/env python
from geometry_msgs.msg import Vector3
import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
import of_library as of


#x: n,2 array of feature pos
#u: n,2 array of feature flows
#d: proposed distance to plain
#TODO move to of_library
def solve_lgs(x,u,d):
    A=np.empty((0,3))
    B=np.empty(0)
    #better would be to do it in parallel
    for i in range(len(x)):
        x_hat=np.array([[0,-1,x[i,1]],[1,0,-x[i,0]],[-x[i,1],x[i,0],0]])
        b_i = np.dot(x_hat,np.append(u[i],0))/np.dot(self.normal,np.append(x[i],1))  #append 3rd dim for calculation (faster method ?)
        A=np.append(A,x_hat,axis=0)
        B=np.append(B,b_i)
    return np.linalg.lstsq(A/d,B)   #v,R,rank,s



class optical_fusion:
    def call_normal(normal):
        self.normal = Vector3()
        #TODO conversion according to normal message type...
        self.normal=normal
        self.got_normal_=True
        return "got normal vector"

    def call_vel(speed):
        self.vel=Vector3()
        self.got_vel_=True

    def call_imu(imu):
        self.ang_vel=imu.angular_acceleration
        self.got_ang_vel_=True

    #camera listener. input: image from ros
    #returns features and flow in px where (0,0) is the top right
    def cal_optical(iamge_raw):
        #parameters--------------------------------------------------
        min_feat= 10  #minimum number of features 
        max_feat=50   #maximum number of features

        #Parameters for corner Detection
        feature_params = dict( qualityLevel = 0.3,
                               minDistance = 20,  #changed from 7
                               blockSize = 32 )  #changed from 7

        # Parameters for lucas kanade optical flow
        lk_params = dict( winSize  = (15,15),   #changed from (15,15)
                          maxLevel = 3,       #changed from 1
                          criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.5)) #changed from 10,0.03
        #------------------------------------------------------------------------------


        image=bridge.imgmsg_to_cv2(image_raw,'bgr8') #TODO convert straight to b/w
        image_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        old_pos=self.feat
        #generate new features if to few where found
        if len(old_pos) <= min_feat: 
            old_pos = np.append(old_pos,
                                cv2.goodFeaturesToTrack(old_gray,mask=None,maxCorners=max_feat-len(old_pos),**feature_params))
        new_pos,status,new_pos_err = cv2.calcOpticalFlowPyrLK(old_gray,image_gray,old_pos,None,**lk_params)
        
        self.feat=new_pos[status==1]
        self.flow=new_pos[status==1]-old_pos[status==1]

        #confirm that a picture has been taken
        self.got_picture_=True 
        return "taken picutre"

 
    def __init__(self):
        #TODO init values in numpy, not true data type
        self.normal = np.array([0,0,1])  #normal vector
        self.vel    = np.array([0,0,1])  #trans vel.
        self.feat   = np.array((0,2))  #array of features
        self.flow   = np.array((0,2))  #array of flows
        self.ang    = Vector3(0,0,0)  #Vector3
        self.d      =  1     #distance in m

        #flags
        self.got_normal_ = False
        self.got_vel_    = False
        self.got_picture_= False
        self.got_ang_vel_= False

        rospy.Subscriber("velocity",Vector3,self.call_vel)  #listener for velocity node
        rospy.Subscriber('/camera/image_raw',Image,self.call_camera)

        while not rospy.is_shutdown():
        #implement flag handling and publishing at certain times
        #solve lgs here -> Time handling ??  
            if self.got_picture_==1:
                #zero picture coordinates before solving lgs
                translation=of.pix_trans((480,640))  
                x=np.array([self.feat[:,0]-translation[0],
                            self.feat[:,1]-translation[1]])
                
                u=self.flow #copy flow
                #calculate feasible points
                #!!Carefull use velocity at time of picture save v_vel in other case !!!
                feasibility,self.d = of.r_tilde(x,u,n,self.vel)
                x=x[feasibility >= T]
                u=u[feasibility >= T]

                #calculate angular vel w from ang_vel of pixhawk (using calibration)
                #TODO!!
                #account for angular vel. (13)
                u=np.array([u[:,0] - x[:,0]*x[:,1]*w[0]+(1+x[:0]**2)*w[1]-x[:,1]*w[2],
                            u[:,1] +(1+x[:,1]**2)*w[0]+x[:,0]*x[:1]*w[1]+x[:,0]*w[2]])

                v_obs,R,rank,s= self.solve_lgs(x,u,self.d)

                #TODO implement kalman filter to combine IMU and optical measurment


if __name__=='__main__':
    rospy.init_node('velocity_calc')
    try:
        node=optical_fusion()
    except rospy.ROSInterruptException: pass

