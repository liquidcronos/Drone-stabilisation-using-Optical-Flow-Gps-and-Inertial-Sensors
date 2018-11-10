import numpy as np
import cv2
import of_library as of
#import rospy
import yaml
import matplotlib.pyplot as plt



#parameter------------------------------
#TODO calculate parameters based on height
max_ft_numb=250        #maximum number of features


feature_params = dict( maxCorners = max_ft_numb,                                    
                     qualityLevel = 0.3,
                      minDistance = 20,  #changed from 7
                        blockSize = 32 )  #changed from 7

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (4,4),   #changed from (15,15)
                  maxLevel = 3,       #changed from 1
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.5)) #changed from 10,0.03

#------------------------------------------------------

#placeholder values for later measurments-------------
d=100  #later calculatet of height measure and dot product
n=np.array([0,0,1])

#-----------------------------------




cap=cv2.VideoCapture("traffic_saigon.avi")

#read first frame and detect features
ret, old_frame=cap.read()
old_gray=cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)  #convert to grayscale


#TODO Tiling to distribute points evenly over image
old_pos = cv2.goodFeaturesToTrack(old_gray, mask =None, **feature_params)

#TODO later includet in new initalisation function
old_pos_err=np.empty((1,2))
#number of features used for tracking
ft_numb=len(old_pos[:,0,0])

#save first feature position
first_pos=old_pos


#initialize d vector
d=np.ones(len(old_pos))*d

#mask to visualize feasible points
mask=np.zeros_like(old_frame)

#initialize Kalman Filter:
kalman=cv2.KalmanFilter(3,3,0)
#A
kalman.transitionMatrix=np.eye(3)
#B
kalman.controlMatrix=np.eye(3)
#C
kalman.measurementMatrix=np.eye(3)

#TODO measurment noise to be later determinend:
kalman.processNoiseCov=1e-5*np.eye(3)
kalman.measurementNoiseCov=1e-1*np.eye(3)
kalman.errorCovPost=0.1*np.eye(3)
#initiate velocity as 0
kalman.statePost=np.zeros(3)
while(1):
    ret,frame = cap.read()
    frame_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #... and convert to gray
    new_pos, status, new_pos_err = cv2.calcOpticalFlowPyrLK(old_gray,frame_gray,old_pos,None,**lk_params) 
    #reshape into proper 2dim array
    new_pos=new_pos.reshape((len(new_pos),2))
    old_pos=old_pos.reshape((len(old_pos),2))

    




    #generate x according to math for simple implementation (not fast or efficient)
    x=np.ones((len(new_pos),3))
    x[:,:-1]=new_pos

    #generate u according to math for simple implementation (not fast or efficient)
    u=np.zeros((len(new_pos),3))
    u[:,:-1]=new_pos-old_pos   #TODO what do we do if we loose features in the meantime?
    #u=u.flatten()



    #TODO calculate u in [m]


    #TODO normally this would use the IMU data and a internal drone modell
    v_new=kalman.predict(np.random.normal(0,0.01,3)).reshape(3)

    # measure how well features fit the assumptions
    feasibility,distance=of.r_tilde(x,u,n,v_new) 
    
    #delete values where row of vectors exede feasibility 
    feasible_new=x[feasibility>=0.5]
    feasible_flow=u[feasibility>=0.5]
    feasible_dist=d[feasibility>=0.5]
    # clusters need to be found and parallel planes need to be tracked ( the difference between d shoudl remain equal) 

    
    #construct LGS to solve for speed using optained d and features    
    A= np.empty((0,3))
    B= np.empty(0)
    for i in range(len(feasible_new)):
        ai=np.array([[0,-1,feasible_new[i,1]],[1,0,-feasible_new[i,0]],[-feasible_new[i,1],feasible_new[i,0],0]])/d[i]
        bi=np.dot(ai,feasible_flow[i])/np.dot(n,feasible_new[i])
        A=np.append(A,ai,axis=0)
        B=np.append(B,bi)
    v_obs,R,rank,s=np.linalg.lstsq(A,B)
    print(v_obs)
    

    #use v_obs for kalman filter correction 
    
    kalman.correct(v_obs)

    v_new=np.dot(kalman.transitionMatrix, v_new)

    #visual test-------------------------------------------
    of.visualize(frame,mask,new_pos[feasibility>=0.3],old_pos[feasibility>=0.3],"test")
    of.visualize(frame,mask,new_pos[feasibility>=0.6],old_pos[feasibility>=0.6],"test",marker=[0,255,0])
    of.visualize(frame,mask,new_pos[feasibility>=0.9],old_pos[feasibility>=0.9],"test",marker=[255,0,0])
    #-----------------------------------------------------------

    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
  
    old_gray=frame_gray.copy()
    old_pos=new_pos
    old_pos_err=new_pos_err

cv2.destroyAllWindows()
cap.release()
