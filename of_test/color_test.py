
# Python program for Detection of a  
# specific color using OpenCV with Python 
import cv2 
import numpy as np  
  
cap = cv2.VideoCapture('traffic_saigon.avi')  

_, oldframe=cap.read()
road=np.ones_like(oldframe[:,:,0])
while(1):        
    # Captures the live stream frame-by-frame 
    _, frame = cap.read()  
    # Converts images from BGR to HSV 
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
    #last value should have the biggest window because of shadows
    lower_red = np.array([0,0,70]) #changed from 110,50,50
    upper_red = np.array([20,20,190]) #changed from 130,255,255
  
    # Here we are defining range of bluecolor in HSV 
    # This creates a mask of blue coloured  
    # objects found in the frame. 
    mask = cv2.inRange(hsv, lower_red, upper_red) 
    
    inversemask=(np.ones_like(mask)*255-mask)/255
    road=inversemask*road
    # The bitwise and of the frame and mask is done so  
    # that only the blue coloured objects are highlighted  
    # and stored in res 
    res = cv2.bitwise_and(frame,frame, mask= mask) 
    cv2.imshow('frame',frame) 
    cv2.imshow('mask',mask) 
    cv2.imshow('res',res) 
    cv2.imshow('road',road*255)
      
    # This displays the frame, mask  
    # and res which we created in 3 separate windows. 
    k = cv2.waitKey(5) & 0xFF
    if k == 27: 
        break
      
# Destroys all of the HighGUI windows. 
cv2.destroyAllWindows() 
  
# release the captured frame 
cap.release() 

