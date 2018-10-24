import numpy as np
import cv2
import of_library as of



def newfeature(bad_points,good_points,immobile):
    good_points=good_points[immobile==True]
    moving_ft=np.ones_like(old_frame) 
    bad_reshaped=bad_points[:,0,:]


    #how to handle moving features:
    #---------------------------------------------------
    '''
    if len(bad_points) >1:
        #crutch which scales clusternumber with point number
        clusternumb=max([3,int(len(bad)/80)])
        #generate clusters of moving features using kmean
        clusterlist=of.kmeancluster(bad_points,clusternumb)
        #drawing bounding boxes around clusters
        of.boundingboxes(clusterlist,moving_ft,feature_params["minDistance"])
    elif len(bad_points) == 1:
        #append to good points to draw circle
        good_points=np.append(good_points,bad_reshaped,axis=0)
    '''
    of.circles(bad_reshaped,moving_ft,feature_params["minDistance"]) #if only using  circles than bad points dont need to be saved



    #how to handle unmoving features
    #--------------------------------------------------------------
    of.circles(good_points,moving_ft,feature_params["minDistance"])



    moving_ft_gray=cv2.cvtColor(moving_ft,cv2.COLOR_BGR2GRAY)

    #drawing mask for debugging
    cv2.imshow('mask',moving_ft_gray*255)
    
    #adding new features
    #---------------------------------------------------------
    new_params=feature_params
    new_params["maxCorners"]=int(ft_numb-np.sum(immobile))
    features=cv2.goodFeaturesToTrack(frame_gray, mask = moving_ft_gray , **new_params)


    #handling case that not enoigh new features where found
    #---------------------------------------------------------------
    missing_feat=int(new_params["maxCorners"]-len(features[:,0,0]))
    if new_params["maxCorners"]!=len(features[:,0,0]):
           print("boop")
           features=np.append(features,1000*np.ones((missing_feat,1,2)),axis=0)



    return features

#parameter------------------------------
#TODO calculate parameters based on height
max_ft_numb=100        #maximum number of features
height=100             #height above ground
#TODO height as array of length max_ft_numb

#TODO generate those from imu and height
max_dist=30 
max_vel=8


feature_params = dict( maxCorners = max_ft_numb,                                    
                     qualityLevel = 0.3,
                      minDistance = 20,  #changed from 7
                        blockSize = 32 )  #changed from 7

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (4,4),   #changed from (15,15)
                  maxLevel = 1,       #changed from 2
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.5)) #changed from 10,0.03
 
#------------------------------------------------------


cap=cv2.VideoCapture("traffic_saigon.avi")
#cv2.VideoCapture(0) for live Camerafeed


#read first frame and detect features
ret, old_frame=cap.read()
old_gray=cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)  #convert to grayscale


#mask for areas excluded from Feature Tracking
#TODO draw Border around image which depends on height
old_pos = cv2.goodFeaturesToTrack(old_gray, mask =None, **feature_params)

#TODO later includet in new initalisation function
old_pos_err=np.empty((1,2))
#number of features used for tracking
ft_numb=len(old_pos[:,0,0])

first_pos=old_pos
bad=np.empty([1,1,2])

#mask for visualization
mask=np.zeros_like(old_frame)
while(1):
    ret,frame = cap.read() #read next frame...
    frame_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #... and convert to gray
    new_pos, status, new_pos_err = cv2.calcOpticalFlowPyrLK(old_gray,frame_gray,old_pos,None,**lk_params)

     
    #select unmoving features
    immobile_points=of.static_immobile(new_pos,old_pos,max_vel,1,[1000,1000])*status
    
    if sum(immobile_points) < ft_numb:
        #generate new features 
        new_points=newfeature(bad,first_pos,immobile_points)
        
        #replace old features with new ones
        new_points=new_points.reshape(-1,2)
        old_pos[immobile_points==False]=new_points
        new_pos[immobile_points==False]=new_points
        first_pos[immobile_points==False]=new_points
         
        #track removed features
        bad=np.append(bad,old_pos[immobile_points==False].reshape(-1,1,2),axis=0)  
        bad=np.append(bad,first_pos[immobile_points==False].reshape(-1,1,2),axis=0)
    
        
        #visualize
        of.visualize(frame,mask,new_pos,old_pos,"frame")


    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
  
    old_gray=frame_gray.copy()
    old_pos=new_pos
    old_pos_err=new_pos_err

cv2.destroyAllWindows()
cap.release()
