#!/usr/bin/env python
from geometry_msgs.msg import Vector3
import rospy
import numpy as np
import cv2
from cv2_bridge import CvBridge CvBridgeError



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
        if len(old_pos) <= min_feat): 
            old_pos = np.append(old_pos,
                                cv2.goodFeaturesToTrack(old_gray,mask=None,maxCorners=max_feat-len(old_pos),**feature_params)
            new_pos,status,new_pos_err = cv2.calcOpticalFlowPyrLK(old_gray,image_gray,old_pos,None,**lk_params)

            self.feat=new_pos
            self.flow=new_pos-old_pos

            #confirm that a picture has been taken
            self.got_picture_=True
        return "taken picutre"




    #x: n,2 array of feature pos
    #u: n,2 array of feature flows
    #d: proposed distance to plain
    def solve_lgs(x,u,d):
        A=np.empty((0,3))
        B=np.empty(0)
        #better would be to do it in parallel
        for i in rajge(len(x))
            x_hat=np.array([[0,-1,x[i,1]],[1,0,-x[i,0]],[-x[i,1],x[i,0],0]])
            b_i = np.dot(x_hat,np.append(u[i],0))/np.dot(self.normal,np.append(x[i],1))  #append 3rd dim for calculation (faster method ?)
            A=np.append(A,x_hat,axis=0)
            B=np.append(B,b_i)
        return np.linalg.lstsq(A/d,B)   #v,R,rank,s

def solve_lgs_server():
    rospy.init_node('solve_lgs_server')
    s = rospy.Service('solve_lgs', LGS, handle_lgs_solving)
    print "Ready to solve LGS"
    rospy.spin()

if __name__ == "__main__":
     solve_lgs_server()


    
    def __init__(self):
        self.normal = #init value  #normal vector
        self.vel    = #init value  #trans vel.
        self.feat   = np.array((0,2))  #array of features
        self.flow   = np.array((0,2))  #array of flows

        #flags
        self.got_normal_ = False
        self.got_vel_    = False
        self.got_pciture_= False

        #features should be class variable...
