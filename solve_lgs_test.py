#!/usr/bin/env python

#simple service to test lgs_solving serice
import rospy
import numpy as np

#client which calls for the service
def solve_lgs_client(x,u,n,d):
    rospy.wait_for_service('add_two_ints')
    try:
        solver = rospy.ServiceProxy('solve_lgs', LGS)
        v_obs,R,rank,s = solver(x,u,n,d)
        return v_obs
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e

def usage():
    return "%s [x y]"%sys.argv[0]

if __name__ == "__main__":
    #testcases go here
    x_test=np.array([[1,1],[2,2]])
    u_test=np.array([[1,1],[1,1]])
    n_test=np.array([0,0,1])
    d_test=1
    print solve_lgs_client(x_test,u_test,n_test,d_test)
