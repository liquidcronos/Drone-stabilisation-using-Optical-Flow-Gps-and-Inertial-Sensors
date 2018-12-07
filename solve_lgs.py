#!/usr/bin/env python

#------------------------------------------------------------------
# ROS NODE which solves LGS according to (14) to calculate speed
#should be used for each plane separatly
# x: n,2 array of feature positions
# u: n,2 array of feature flows
# n: 3d normal vector (normalized g vector)
# d: proposed distance to plain
#-----------------------------------------------------------------
import rospy
import numpy as np


def handle_lgs_solving(x,u,n):
	#init empty matrix and vector
	A=np.empty((0,3))
	B=np.empty(0)
    #should work in parallel
    for i in range(len(x)):
		x_hat=np.array([[0,-1,x[i,1]],[1,0,-x[i,0]],[-x[i,1],x[i,0],0]])
		b_i = np.dot(x_hat,np.append(u[i],0))/np.dot(n,np.append(x[i],1))  #append 3rd dim for calculation (faster method ?)
		A=np.append(A,x_hat,axis=0)
		B=np.append(B,b_i)
    v_obs,R,rank,s=np.linalg.lstsq(A/d,B)
    return solve_lgs_response(v_obs,R)   #also return rank?

def solve_lgs_server():
    rospy.init_node('solve_lgs_server')
    s = rospy.Service('solve_lgs', LGS, handle_lgs_solving)
    print "Ready to solve LGS"
    rospy.spin()

if __name__ == "__main__":
     solve_lgs_server()


