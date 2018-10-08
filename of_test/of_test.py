import numpy as np
import cv2





def tomuchvel(newpos,oldpos,maxspeed,dist_to_cam):
    immobile= (np.abs(newpos-oldpos)<(maxspeed/dist_to_cam))
    return immobile[:,:,1]*immobile[:,:,0]
    

def findcluster(bad,numb_of_cluster):

    #generate empty list for each cluster
    clusterlist=[]

    #find clusters using kmean
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    bad=np.float32(bad)
    ret,label,center=cv2.kmeans(bad,numb_of_cluster,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
           
    #seperate into different cluster
    for i in range(numb_of_cluster):
        newcluster=bad[label.ravel()==i]
        clusterlist.append(newcluster)
    #return array of clusters
    return(np.array([np.array(cluster) for cluster in clusterlist]))


#for c++ opencv includes this function
def hierachicalcluster(bad):
    return("get this shit to work with python")


cap = cv2.VideoCapture('traffic_saigon.avi')
#cap = cv2.VideoCapture('highway.avi')

#--------------------------------------------------------------------------------------
# params for ShiTomasi corner detection
max_ft_numb=10              #maximum number of corners

max_dist=30                  #maximum distance before cutting feature
max_vel = 8                  #maximum velocity before cutting feature

feature_params = dict( maxCorners = max_ft_numb,
                       qualityLevel = 0.3,
                       minDistance = 20,  #changed from 7
                       blockSize = 32 )  #changed from 7
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (4,4),   #changed from (15,15)
                  maxLevel = 1,       #changed from 2
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.5)) #changed from 10,0.03




#------------------------------------------------------------------------------------


# Create some random colors
color = np.random.randint(0,255,(max_ft_numb*2,3))  #*2 because we need more colors for new trails

    

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
bad_ft_mask=np.ones_like(old_frame)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None ,useHarrisDetector=False, **feature_params)
#TODO there should be a mask that excludes the outer perimiter of the image as those points are not fit for tracking

#remember first position
pfirst=p0
#initialize list of rejected features
bad = np.empty([1,2])
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)





while(1):
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # calculate new position
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
  
     
    #classification of "good" features 
    good     = tomuchvel(p1,pfirst,max_dist,1)
    good_vel = tomuchvel(p1,p0,max_vel,1)     #TODO use height estimation


    
    # Select good points
    good_new   =     p1[good_vel*good*st==True] #and good==1
    good_old   =     p0[good_vel*good*st==True]  #normal st==1
    good_first = pfirst[good_vel*good*st==True]

    bad = np.append(bad,pfirst[good_vel*good*st==False],axis=0)
    bad = np.append(bad,p1[good_vel*good*st==False],axis=0)
    print(len(bad))


    # draw the tracks 
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1) 
        img = cv2.add(frame,mask)
        cv2.imshow('frame',img)
  
        #cv2.imshow('edge',cv2.Canny(img,100,200))

    #find new features if to few:++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if len(p1)< 10:   #sets minimum number of features 
        clusterlist =findcluster(bad,3)
       
       
        
        for clusters in clusterlist:
            
            #draw axis aligned bounding boxes:    
            '''
            x,y,w,h=cv2.boundingRect(clusters)

            #generate a mask for the new features
            bad_ft_mask=cv2.rectangle(bad_ft_mask,(x,y),(x+w,y+h),0,thickness=cv2.FILLED)
            '''
            
            #draw rotaded bounding box:
            rect=cv2.minAreaRect(clusters)
            box=cv2.boxPoints(rect)
            box=np.int0(box)
            cv2.drawContours(bad_ft_mask,[box],0,0,cv2.FILLED)
            
            #generate mask using simple circles
            '''
            for points in clusters:
                x=int(points[0])
                y=int(points[1])
                bad_ft_mask=cv2.circle(bad_ft_mask,(x,y),20,0,thickness=cv2.FILLED)  
            '''
            
            #fill cluster wich convex hull
            '''
            filler = cv2.convexHull(clusters,returnPoints=True)
            filler=np.array(filler,dtype='int32')
            bad_ft_mask=cv2.fillConvexPoly(bad_ft_mask,filler,0)
            '''





        #exclude already used featuers:
        for  points in good_first:
            x=int(points[0])
            y=int(points[1])
            bad_ft_mask=cv2.circle(bad_ft_mask,(x,y),feature_params["minDistance"],0,thickness=cv2.FILLED)  
        
        #convert to grayscale for mask
        bad_ft_mask_gray = cv2.cvtColor(bad_ft_mask, cv2.COLOR_BGR2GRAY)

        #draw new boundingbox
        cv2.imshow('mask',bad_ft_mask*255) 

        #add new features with updatet mask:
        new_features=cv2.goodFeaturesToTrack(old_gray, mask = bad_ft_mask_gray ,useHarrisDetector=False, **feature_params)
        #TODO only generate enough points to get back to 100 features
        good_new = np.append(good_new,new_features)
        good_old = np.append(good_old,new_features)
        good_first = np.append(good_first,new_features)
        
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


        

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)
    pfirst=good_first.reshape(-1,1,2)


cv2.destroyAllWindows()
cap.release()
