import cv2
import yaml
import rospy
from cv_bridge import CvBridge
import numpy as np

bridge =CvBridge()
streamCam=file('camData.yaml', 'r')
dataCam=yaml.load(streamCam)
cv2_image=[]
for data in dataCam:
    cv2_image =bridge.compressed_imgmsg_to_cv2(data)
    cv2.imshow('frame',cv2_image)
    cv2.waitKey(50)
