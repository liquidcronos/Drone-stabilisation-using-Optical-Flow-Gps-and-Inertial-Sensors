#Library for Feature and Optical Data related functions

import numpy as np
import cv2


#visualization function 
#plots current position as circle and leaves Trail over previous positions.
#################################################
#image: camera image on which to draw           #
#marker: array which stores trail               #
#newpos: new position of features               #
#oldpos: old position of features               #
#marker: color to draw in rgb (default=red)     #
#frame_name: name of window                     #
#################################################

def visualize(image,mask,newpos,oldpos,frame_name="visualization",marker=[0,0,255],):

    for i,(new,old) in enumerate(zip(newpos,oldpos)):
        a,b=new.ravel()
        c,d=old.ravel()
        mask=cv2.line(mask, (a,b), (c,d), marker, 2) #local assignement not global in c++
        image=cv2.circle(image,(a,b),5,marker,-1)
        img=cv2.add(image,mask)
        cv2.imshow(frame_name,img)





#function to determine stable features
#returns True for stable and False for unstable
############################################################
#newpos: new position of features                          #
#oldpos: old position of features                          #
#maxspeed: maximum permitted velocity in world system      #
#distance: height over feature (pinhole camera modell)     #
#dummy_valie: value asigned if point is no longer viable   #
############################################################
def immobile(newpos,oldpos,maxspeed,distance,dummy_value):
    speed_constraint = (np.abs(newpos-oldpos))<(maxspeed/distance)  #True if velocity is less than maxspeed
    dummy_constraint = (oldpos) != dummy_value #or (newpos) != dummy_value #True if value is not dummy
    stable=speed_constraint*dummy_constraint
    return stable[:,:,0]*stable[:,:,1]





#returns array of arrays where each array is a cluster using kmean clustering
#####################################
#points: points to sort in cluster  #
#k : number of clusters             #
#####################################
def kmeancluster(points,k):

      #generate empty list for each cluster
      clusterlist=[]

      #find clusters using kmean
      criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
      points=np.float32(points)
      ret,label,center=cv2.kmeans(points,k,None,criteria    ,10,cv2.KMEANS_RANDOM_CENTERS)

      #seperate into different cluster
      for i in range(k):
          newcluster=points[label.ravel()==i]
          clusterlist.append(newcluster)
      #return array of clusters
      return(np.array([np.array(cluster) for cluster in clusterlist    ]))


#clustering based on distance of points to each other. 
#already established clusterlist (in indices)
def distancecluster(pointcloud,points,maxdist,clusterlist):
    for point in points:
        fusion=[]    #TODO create truly empty array
        for cluster in clusterlist:
            part_of_cluster=np.sum([np.abs(clusterpoints-point) <= maxdidst for clusterpoints in cluster])
            if part_of_cluster ==0:
                fusion=np.append(newlist,cluster)
        if fusion != []:
           newlist=clusterlist[fusion != clusterlist] 
           newlist=np.abbend(newlist,np.concatenate(fusion))
        else:
           newlist=np.array([point])
    return newlist



      


#generate rotated boundingbox mask over each cluster
#if cluster containes only one point draw a circle of given radius
################################################################
#clusterlist: array of arrays (see kmeanclustering output)     #
#mask: array on which to draw on                               #
#radius: radius of circle if clusters contains only one point  #
################################################################
def boundingboxes(clusterlist,mask,radius):
    for cluster in clusterlist:
        if len(cluster) >1:
            rect=cv2.minAreaRect(cluster)
            box=cv2.boxPoints(rect)
            box=np.int0(box)
            cv2.drawContours(mask,[box],0,0,cv2.FILLED)
        elif len(cluster) ==1:
            circles(cluster[:,0,:],mask,radius)





#generated circle mask of given radius around each point
################################################################
#points: points to draw circles around                         #      
#mask: array on which to draw on                               #
#radius: radius of circle if clusters contains only one point  #
################################################################
def circles(points,mask,radius):
    for point in points:
        x=int(point[0])
        y=int(point[1])
        cv2.circle(mask,(x,y),radius,0,cv2.FILLED)



#generate convex hull mask for each cluster
#if cluster containes only one point draw a circle of given radius
################################################################
#clusterlist: array of arrays (see kmeanclustering output)     #
#mask: array on which to draw on                               #
#radius: radius of circle if clusters contains only one point  #
################################################################
def convexhull(clusterlist,mask,radius):
    for cluster in clusterlist:
        if len(cluster) >1:
            filler = cv2.convexHull(cluster,returnPoints=True)
            filler=np.array(filler,dtype='int32')
            cv2.fillConvexPoly(mask,filler,0)
        elif len(cluster)==1:
            circles(cluster[0,:,:],mask,radius)


# initializes the optical flow and searches for best points to keep on tracking
#endcount: how many features to return
def initialize_ft(camera,feature_parameter,lk_parameter,iterations,end_count):

    #initialize 1st Frame
    cap=cv2.VideoCapture(camera)
    ret, old_frame=cap.read()
    old_gray=cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    old_pos = cv2.goodFeaturesToTrack(old_gray, mask =None, **feature_parameter)
   
    if end_count <= 0:
        raise ValueError(' end_count must be a positive number')
    if iterations <= 0:
        raise ValueError(' iterations must be a positive number')
    
    for i in range(iterations):
        ret,frame = cap.read() #read next frame...
        frame_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #... and convert to gray
        new_pos, status, error = cv2.calcOpticalFlowPyrLK(old_gray,frame_gray,old_pos,None,**lk_params)


        #select unmoving features
        immobile_points=immobile(new_pos,old_pos,max_vel,1,[1000,1000])*status

        #TODO establish metric for stable points via paralax


