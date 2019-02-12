import numpy as np
import cv2
import of_library as of
#import rospy
import yaml
import matplotlib.pyplot as plt



#parameter------------------------------
#TODO calculate parameters based on height
max_ft_numb=50        #maximum number of features


feature_params = dict( maxCorners = max_ft_numb,                                    
                     qualityLevel = 0.3,
                      minDistance = 20,  #changed from 7
                        blockSize = 32 )  #changed from 7

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),   #changed from (15,15)
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

#initialize treshhold
T=0.9
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
kalman.measurementNoiseCov=1e1*np.eye(3)
kalman.errorCovPost=0.1*np.eye(3)
#initiate velocity as 0
kalman.statePost=np.zeros(3)

while(1):
    ret,frame = cap.read()
    frame_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #... and convert to gray

    #generate new features if to many where lost (probably needs to be smarter
    if (len(old_pos) <= 10):
        new_params=feature_params
        new_params["maxCorners"]=int(ft_numb-len(old_pos))
        #TODO should append features
        old_pos = cv2.goodFeaturesToTrack(old_gray, mask =None, **new_params)

    new_pos, status, new_pos_err = cv2.calcOpticalFlowPyrLK(old_gray,frame_gray,old_pos,None,**lk_params) 
    print("relative error:", np.mean(new_pos_err/new_pos))
    #reshape into proper 2dim array
    new_pos = new_pos.reshape((len(new_pos),2))
    old_pos = old_pos.reshape((len(old_pos),2))
    status=status.reshape(len(status))

    #generate x according to math for simple implementation (not fast or efficient)
    x        = np.ones((len(new_pos),3))
    x[:,:-1] = new_pos

    # (0,0) at center of image
    translation=of.pix_trans((480,640))
    x[:,0]=x[:,0]-translation[0]
    x[:,1]=x[:,1]-translation[1]


    #generate u according to math for simple implementation (not fast or efficient)
    #TODO u need to be purged of rotational component
    u         = np.zeros((len(new_pos),3))
    u[:,:-1]  = new_pos-old_pos


    #random angular momentum
    w=np.random.normal(0,0.01,3)

    #account for angular velocity
    u[:,0] = new_pos[:,0]*new_pos[:,1]*w[0]+(1+new_pos[:,0]**2)*w[1]-new_pos[:,1]*w[2]
    u[:,1] = -(1+new_pos[:,1]**2)*w[0]+new_pos[:,0]*new_pos[:,1]*w[1]+new_pos[:,0]*w[2]

    #TODO calculate u in [m]


    #TODO normally this would use the IMU data and a internal UAV model
    v_new = kalman.predict(np.random.normal(0,0.01,3)).reshape(3)

    # measure how well features fit the assumptions
    feasibility,distance = of.r_tilde(x,u,n,v_new) 

    print(feasibility,distance)
    #delete values where row of vectors exede feasibility 
    feasible_new  = x[feasibility-(status-1)>=T]
    feasible_flow = u[feasibility-(status-1)>=T]
    feasible_dist = distance[feasibility-(status-1)>=T]
    # clusters need to be found and parallel planes need to be tracked ( the difference between d shoudl remain equal) 

    
    #construct LGS to solve for speed using optained d and features    
    A = np.empty((0,3))
    B = np.empty(0)
    if len(feasible_new)<=3:
        continue
    for i in range(len(feasible_new)):
        ai = np.array([[0,-1,feasible_new[i,1]],[1,0,-feasible_new[i,0]],[-feasible_new[i,1],feasible_new[i,0],0]])/feasible_dist[i]
        bi = np.dot(ai,feasible_flow[i])/np.dot(n,feasible_new[i])
        A  = np.append(A,ai,axis=0)
        B  = np.append(B,bi)
    print(A.shape,B.shape)
    v_obs,R,rank,s = np.linalg.lstsq(A,B)
    print('------------------')
    print(v_obs)
    print(v_new)

    #use v_obs for kalman filter correction 
    v_new=kalman.correct(-v_obs) 
    print(v_new)
    print('------------------')

    #visual test-------------------------------------------
    of.visualize(frame,mask,new_pos[feasibility>=T],old_pos[feasibility>=T],"test",marker=[255,0,0])
    #-----------------------------------------------------------
 
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break


    old_gray=frame_gray.copy()
    old_pos=new_pos[feasibility-(status-1)>=T]
    old_pos_err=new_pos_err[feasibility-(status-1)>=T]    

cv2.destroyAllWindows()
cap.release()
