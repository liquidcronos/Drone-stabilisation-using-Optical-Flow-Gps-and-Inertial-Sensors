#Library for Feature and Optical Data related functions

import numpy as np
import cv2
import yaml


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




#converts speed info and feature info to excpected of
###################################################
#position: position of feature                    #
#speed: speed of moving UAV [x,y] or [x,y,z]      #
#height: height of each feature                   #
#focal_len: focal len of camera                   #
#img_dim: dimensions of camera image in px        #
###################################################
def convert_to_of(pos,pos_err,speed,speed_err,height,height_err,focal_len,img_dim):

    #check if division by zero could occur
    if np.any(height < eps): 
        raise ValueError(' height over feature is Zero or Negative')

    # check if dimensions are divisible by 2 and calculate translation
    if img_dim[0]%2 ==0:
        trans_x=img_dim[0]/2
    else:
        trans_x=(img_dim[0]+1)/2

    if img_dim[1]%2 ==0:
        trans_y=img_dim[1]/2
    else:
        trans_y=(img_dim[1]+1)/2

    #expected flow:
    x_exp=(focal_len-pos[0,:]+trans_x)*speed[0]/height
    y_exp=(focal_len-pos[1,:]+trans_y)*speed[1]/height 
    of_exp=[x_exp,y_exp]
    
    #square of std.
    x_exp_err= (pos_er[0,:]*speed[0]/height)**2+((focal_len-pos[0,:]+trans_x) \
            *speed_err[0]/height)**2+((focal_len-pos[0,:]+trans_x)*speed[0]*height_err/height**2)**2
    y_exp_err= (pos_err[1,:]*speed[1]/height)**2+((focal_len-pos[1,:]+trans_y) \
            *speed_err[1]/height)**2+((focal_len-pos[1,:]+trans_y)*speed[1]*height_err/height**2)**2
    of_exp_err=[x_exp_err,y_exp_err]

    return of_exp, of_exp_err


#function to determine stable features
#returns True for stable and False for unstable
#uses static constrains instead of IMU and GPS 
############################################################
#newpos: new position of features                          #
#oldpos: old position of features                          #
#maxspeed: maximum permitted velocity in world system      #
#distance: height over feature (pinhole camera modell)     #
#dummy_valie: value asigned if point is no longer viable   #
############################################################
def static_immobile(newpos,oldpos,maxspeed,distance,dummy_value):
    speed_constraint = (np.abs(newpos-oldpos))<(maxspeed/distance)  #True if velocity is less than maxspeed
    dummy_constraint = (oldpos) != dummy_value #or (newpos) != dummy_value #True if value is not dummy
    stable=speed_constraint*dummy_constraint
    return stable[:,:,0]*stable[:,:,1]



#dynamicly checks for stable features using drone velocity in world system and pin-hole modell
################################################################
#dummy_value: value asigned to points which are not to be used #
################################################################
def dynamic_immobile(newpos,newpos_err,oldpos,oldpos_err,speed,speed_err,focal_len,dummy_value,height,height_err,img_dim):
    
    #observed velocity
    of_obs=newpos-oldpos
    of_obs_err=oldpos_err**2+newpos_err**2   #square of std.

    #expected velocity
    of_exp, of_exp_err = convert_to_of(new_pos,new_pos_err,speed,speed_err,height,height_err,focal_len,img_dim)


    speed_constraint= ((of_obs-of_exp)**2)<(of_obs_err+of_exp_err)
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
##########################################################################
#clusterlist: list of lists containing the indices of each clusterpoint  #
#point_cloud: array of points already used                               #
#points: new points to be clustered                                      #
#maxdist: max distance for clusterpoints                                 #
##########################################################################
def distancecluster(pointcloud,points,maxdist,clusterlist):
    for i in range(len(points)):
        point=points[i]
        #get indices of points that are near enough for clustering
        to_fuse=np.where(np.abs(pointcloud-point) <maxdist)

        clusters_to_fuse=[]
        #check if any such elements are found
        if len(to_fuse) != 0:
            for i  in range(len(clusterlist)):
                #check if cluster contains relevant points
                if np.sum(np.isin(clusterlist[i],to_fuse)) != 0:
                    clusters_to_fuse.append(i)

            #next fuse those clusters
            new_cluster=np.append(clusterlist[clusters_to_fuse])
            new_cluster=np.append(new_cluster,len(pointcloud)+i+1)
            #delete the old clusters
            del clusterlist[clusters_to_fuse]
            #add fusion of clusters
            clusterlist.append(new_cluster)

        else:
            clusterlist.append(len(pointcloud)+i+1)

    return clusterlist,np.append(pointcloud,points)






      


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
#vel: velocity in drone frame
def initialize_ft(camera,feature_parameter,lk_parameter,iterations,end_count,vel,vel_err,focal_len,dummy_value,img_dim):

    #initialize 1st Frame
    cap=cv2.VideoCapture(camera)
    ret, old_frame=cap.read()
    old_gray=cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    #TODO initalize mask
    old_pos = cv2.goodFeaturesToTrack(old_gray, mask =None, **feature_parameter)
   
    if end_count <= 0:
        raise ValueError(' end_count must be a positive number')
    if iterations <= 0:
        raise ValueError(' iterations must be a positive number')
    
    for i in range(iterations):
        ret,frame = cap.read() #read next frame...
        frame_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #... and convert to gray
        new_pos, status, new_pos_err = cv2.calcOpticalFlowPyrLK(old_gray,frame_gray,old_pos,None,**lk_params)
        
        #calculate height once in first step (inprecise)
        #TODO get a better height evaluation (using Kalmann?)
        if i == 0:
            height,height_err=calc_height(new_pos-old_pos,new_pos_err,vel,vel_err,focal_len,new_pos,newpos_err)


        #select unmoving features
        immobile_points=dynamic_immoblie(new_pos,new_pos_err,old_pos,old_pos_err,vel,vel_err,focal_len,dummy_value,height,height_err,img_dim)*status
     

        #TODO establish metric for stable points via paralax
        #TODO what happens if not enough points where found ?
        #TODO return good points and list of bad points for mask



#calculates feature height from optical flow and imu data using pin hole model:
def calc_height(of,of_eff,vel,vel_err,focal_len,newpos,newpos_err):
    
    #calculate height using pin hole modell for both dimensions
    height_x=(focal_len*vel[:,0]-newpos[:,0]*vel[:,2])/of[:,0]
    height_y=(focal_len*vel[:,1]-newpos[:,1]*vel[:,2])/of[:,1]
    
    #square of std.
    height_x_err= (focal_len*vel_err[:,0]/of[:,0])**2+((focal_len*vel[:,0]*of_err[:,0]-new_pos[:,0]*vel[:,2]*of_err[:,0])/of_err[:,0]**2)**2 \
            +(new_pos_err[:,0]*vel[:,2]/of[:,0])**2+(new_pos[:,0]*vel_err[:,2]/of[:,0])**2
    height_y_err= (focal_len*vel_err[:,1]/of[:,1])**2+((focal_len*vel[:,1]*of_err[:,1]-new_pos[:,1]*vel[:,2]*of_err[:,1])/of_err[:,1]**2)**2 \
            +(new_pos_err[:,1]*vel[:,2]/of[:,1])**2+(new_pos[:,1]*vel_err[:,2]/of[:,1])**2

    height=0.5*(height_x+height_y)
    #square of std.
    height_err=height_x_err+heigh_y_err

    return height, height_err


#evaluates feature for use of feature detection:
#def evaluate_ft(


#read acceleration values from yaml file and generate new object for testing
#returns stack of vectors containing the measurments:
def read_yaml_imu(yamlfile):
    streamData=file(yamlfile,'r')
    imuData = yaml.load(streamData)
    
    #starts with last element and builds stack 
    datastack=[]
    for entry in reversed(imuData):
        seconds=entry.header.stamp.secs
        nseconds=entry.header.stamp.nsecs

        orientation=entry.orientation
        orientation=[orientation.x,orientation.y,orientation.z,orientation.w]
        orientation_err=entry.orientation_covariance

        linear_acc= entry.linear_acceleration
        linear_acc=[linear_acc.x,linear_acc.y,linear_acc.z]
        linear_acc_err=entry.linear_acceleration_covariance
        
        angular_vel=entry.angular_velocity
        anuglar_vel=[angular_vel.x,angular_vel.y,angular_vel.z]
        angular_vel_err=entry.angular_velocity_covariance
       
        #TODO nseconds roundet..maybe also start seconds at 0
        datastack.append([seconds+float(nseconds/10**6),orientation,orientation_err,linear_acc,linear_acc_err,angular_vel,angular_vel_err]) 
    return datastack
    
    
    

#calculates velocity of drone in realive space.
#def calculate_speed(orientation,orientation_err,angular_acc,angular_acc_err,linear_acc,linear_acc_err):
