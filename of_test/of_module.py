import numpy as np
import cv2


def immobile(newpos,oldpos,firstpos,maxspeed,maxdist):
    speed_const= (np.abs(newpos-oldpos))<(maxspeed)
    dist_const = (np.abs(newpos-firstpos))<(maxdist)
    return (speed_const*dist_const)[:,:,1]*(speed_const*dist_const)[:,:,0]




#old clusterfinding module
def findcluster(bad,numb_of_cluster):
                                                                  
     #generate empty list for each cluster
     clusterlist=[]
 
     #find clusters using kmean
     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
     bad=np.float32(bad)
     ret,label,center=cv2.kmeans(bad,numb_of_cluster,None,criteria    ,10,cv2.KMEANS_RANDOM_CENTERS)
           
     #seperate into different cluster
     for i in range(numb_of_cluster):
         newcluster=bad[label.ravel()==i]
         clusterlist.append(newcluster)
     #return array of clusters
     return(np.array([np.array(cluster) for cluster in clusterlist    ]))


def newfeature(bad_points,good_points):
    if len(bad_points) >1:
        clusterlist=findcluster(bad_points,3)
       
        #drawing bounding boxes around moving clusters
        for cluster in clusterlist:
            rect=cv2.minAreaRect(cluster)
            box=cv2.boxPoints(rect)
            box=np.int0(box)
            cv2.drawContours(moving_ft,[box],0,0,cv2.FILLED)
    else:
        print(good_points.shape)
        bad_reshaped=bad_points[:,0,:]
        good_points=np.append(good_points,bad_reshaped,axis=0)
    #drawing circles around immobile points
    for points in good_points:
        x=int(points[0])
        y=int(points[1])
        cv2.circle(moving_ft,(x,y),feature_params["minDistance"],0,cv2.FILLED)
    
    #adding new features
    moving_ft_gray=cv2.cvtColor(moving_ft,cv2.COLOR_BGR2GRAY)
    

    cv2.imshow('mask',moving_ft_gray*255)
    new_params=feature_params
    new_params["maxCorners"]=int(10-np.sum(immobile_points))
    print(new_params["maxCorners"])
    return cv2.goodFeaturesToTrack(frame_gray, mask = moving_ft_gray , **new_params)

#parameter------------------------------
#TODO calculate parameters based on height
max_ft_numb=10         #maximum number of features
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
moving_ft=np.ones_like(old_frame) 
#TODO draw Border around image which depends on height

old_pos = cv2.goodFeaturesToTrack(old_gray, mask =None, **feature_params)

first_pos=old_pos

bad=np.empty([1,1,2])
while(1):
    ret,frame = cap.read() #read next frame...
    frame_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #... and convert to gray
    cv2.imshow("frame",frame_gray)
    new_pos, status, error = cv2.calcOpticalFlowPyrLK(old_gray,frame_gray,old_pos,None,**lk_params)

     
    #select unmoving features
    immobile_points=immobile(new_pos,old_pos,first_pos,max_vel,max_dist)*status
    
    #generate new features
    if sum(immobile_points) < 10:
        new_points=newfeature(bad,first_pos[immobile_points*status==True])
        

        #replace old features with new ones
        new_points=new_points.reshape(-1,2)
        old_pos[immobile_points==False]=new_points
        new_pos[immobile_points==False]=new_points
        first_pos[immobile_points==False]=new_points
         
        #track moving feautres
        bad=np.append(bad,old_pos[immobile_points==False].reshape(-1,1,2),axis=0)  
        bad=np.append(bad,first_pos[immobile_points==False].reshape(-1,1,2),axis=0)
    

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

   
    old_gray=frame_gray.copy()
    old_pos=new_pos

cv2.destroyAllWindows()
cap.release()
