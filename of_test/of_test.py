import numpy as np
import cv2



#function wich returns true for points inside convex hull of given cluster
def in_hull(point,cluster):
    #generate convex hull
    hull=cv2.convexHull(cluster)
   
    #check if point is in hull
    inhull=cv2.pointPolygonTest(hull,point,measureDist=False)

    if inhull >=0:
        return True
    else:
        return False

    



#function wich returns true for points true for points inside bounding box
# much simpler computation bu not as precise as bounding box
def in_boundingbox(point,cluster):
    
    #(x,y) top left corner, w =width, h=height
    x,y,w,h=cv2.boundingRect(cluster)

    if (x <= point[0] <= x+w) and ( y <= point[1] <= y+h):
        return True
    else:
        return False
    #TODO get boundingbox coordinates






def displsum(flow,firstpos,distance,distsum):
    rel_pos=flow-firstpos
    distsum=distsum+rel_pos
    immobile=(distsum<distance)
    return immobile[:,:,1]*immobile[:,:,0],distsum

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
        
    
    #color for each cluster
    colorcl = np.random.randint(0,255,(numb_of_cluster,3))
    
    #seperate into different cluster
    for i in range(numb_of_cluster):
        newcluster=bad[label.ravel()==i]
        clusterlist.append(newcluster)
    #return array of clusters
    return(np.array([np.array(cluster) for cluster in clusterlist]))




cap = cv2.VideoCapture('traffic_saigon.avi')
#cap = cv2.VideoCapture('highway.avi')

#--------------------------------------------------------------------------------------
# params for ShiTomasi corner detection
max_ft_numb=100               #maximum number of corners

max_dist=30                  #maximum distance before cutting feature
max_vel = 8                  #maximum velocity before cutting feature

feature_params = dict( maxCorners = max_ft_numb,
                       qualityLevel = 0.3,
                       minDistance = 4,  #changed from 7
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


#testmask=np.zeros_like(old_gray)
#testmask[200:300,200:300]=1

p0 = cv2.goodFeaturesToTrack(old_gray, mask = None ,useHarrisDetector=False, **feature_params)
#TODO there should be a mask that excludes the outer perimiter of the image as those points are not fit for tracking


pfirst=p0

bad = np.empty([1,2])
distsum=np.zeros((len(p0),1,2))
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)





while(1):
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
  
    
   
  
      
    #classification of "good" features 
    good=tomuchvel(p1,pfirst,max_dist,1)
    #good,distsum=displsum(p1,p0,max_dist,distsum)
    good_vel = tomuchvel(p1,p0,max_vel,1)     #TODO use height estimation


    #number of features tracked
    #print(len(p0))
    
    # Select good points
    good_new   =     p1[good_vel*good*st==True] #and good==1
    good_old   =     p0[good_vel*good*st==True]  #normal st==1
    good_first = pfirst[good_vel*good*st==True]
    #good_sum   = distsum[good*st==True]

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
   

    #find new features if to few:++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if len(p1)<= 70:   #sets minimum number of features 
        clusterlist =findcluster(bad,4)
       
        #TODO provide a mask for the feature detection with the convex hull of each cluster cut out
        
        for clusters in clusterlist:
            '''
            for x in range(old_gray.shape[0]):
                for y in range(old_gray.shape[1]):
                         if in_boundingbox(np.array([x,y]),clusters):   #TODO inefficient as box is computed each time
                             bad_ft_mask[x,y]=0
            '''
            #draw boundin boxes:    
            x,y,w,h=cv2.boundingRect(clusters)
            
            
            #generate a mask for the new features
            bad_ft_mask=cv2.rectangle(bad_ft_mask,(x,y),(x+w,y+h),0,thickness=cv2.FILLED)
            #draw new boundingbox



        bad_ft_mask_gray = cv2.cvtColor(bad_ft_mask, cv2.COLOR_BGR2GRAY)
        #exclude already used featuers:
        for  points in good_first:
            x=int(points[0])
            y=int(points[1])
            bad_ft_mask=cv2.circle(bad_ft_mask,(x,y),7,0,thickness=cv2.FILLED)  #7 is min distance between features
        
        cv2.imshow('mask',bad_ft_mask*255) 
        #add new features with updatet mask:
        new_features=cv2.goodFeaturesToTrack(old_gray, mask = bad_ft_mask_gray ,useHarrisDetector=False, **feature_params)
        good_new = np.append(good_new,new_features)
        good_old = np.append(good_old,new_features)
        good_first = np.append(good_first,new_features)
        
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    '''


        
        #draw clusters for easy debugging------------------------------------------------------------------------
        #iterate over all clusters... 
        masks=np.empty(len(clusterlist),dtype=object)
        for i in range(len(clusterlist)): 
            features=clusterlist[i]
            #mask to draw clusters on
            clustermask=np.zeros_like(old_frame)
            
            

            #... and draw each point for each cluster 
            for j in range(len(features)):  
                masks[i] = cv2.circle(clustermask,(features[j,0],features[j,1]),5,color[i].tolist(),-1)  
                #cv2.imshow('cluster_image'+str(i),cv2.add(frame,clustermask))

        cv2.imshow('cluster_image',cv2.add(frame,masks[1])) #change [1] for different clusters
        
        #---------------------------------------------------------------------------------------------------------
    '''     
        

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)
    pfirst=good_first.reshape(-1,1,2)
    #distsum=good_sum.reshape(-1,1,2)


cv2.destroyAllWindows()
cap.release()
