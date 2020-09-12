#!/usr/bin/env python

## Simple listener that listens to std_msgs/Strings published 
## to the 'listener_cam' topic
from __future__ import print_function
import rospy
from sensor_msgs.msg import Imu
import yaml
import numpy as np
from rospy.numpy_msg import numpy_msg
import  time


import multiprocessing

streamImu = open('imuData.yaml','w')




def callback_Imu(data):

	dumpdata = [data]
	yaml.dump(dumpdata,streamImu)
def listener_imu():
	#print('imu node')
	rospy.init_node('listener_imu', anonymous=True)
	rospy.Subscriber('/mavros/imu/data', Imu, callback_Imu)
	rospy.spin()
	
if __name__ == '__main__':
	print ("starting data capture")
	stop_time = time.time() + 120
	p_imu = multiprocessing.Process(target = listener_imu)
	p_imu.start()
	while stop_time > time.time():
		pass
	print ("killing processes")
	p_imu.terminate()
	p_cam.terminate()
	print ("all worked correctly")
	
	
