# -*- coding: utf-8 -*-
import time
import numpy as np
import matplotlib.pyplot as plt
from  scipy.signal import find_peaks_cwt

def generate_test_data(x,v,omega,d,n,t):
    flow=np.zeros((len(x),3))
    v=v+np.cross(omega,t)
    for i in  range(len(x)):
        flow[i]=np.dot(n,np.append(x[i],1.0))/d*(v-v[2]*np.append(x[i],1.0))+(np.cross(omega,np.append(x[i],1.0))-np.cross(omega,np.append(x[i],1.0))[2]*np.append(x[i],1.0))
    return flow[:,:2]


def solve_lgs(x,u,d,n,omega,t):
    A=np.empty((0,3))
    B=np.empty(0)
   #better would be to do it in parallel
    for i in range(len(x)):
        x_hat=np.array([[0,-1.0,x[i,1]],[1.0,0,-x[i,0]],[-x[i,1],x[i,0],0]])
        b_i = np.dot(x_hat,np.array([u[i,0],u[i,1],0])+1.00*np.dot(x_hat,omega))  #append 3rd dim for calculation (faster method ?)
        A=np.append(A,x_hat*np.dot(n,np.array([x[i,0],x[i,1],1.0]))  ,axis=0)
        B=np.append(B,b_i)
        #'''
        
    try:
        v,R,rank,s = np.linalg.lstsq(A,B*d)   #v,R,rank,s
        return v-np.cross(omega,t),R,s
    except:
        return np.zeros(3), 10000 * np.ones(3*len(x)),np.array([0,0,0])





def of_simulation(linear_velocity,angular_velocity,height_above_gr,normal_vector,translation,pos,ang_vel_sig,translation_sig,height_sig,flow_sig,position_sig,normal_sig):
    v_obs=np.zeros((iterations,3))
    R    =np.zeros(iterations)
    for i in range(iterations):
        ang_vel_err        = angular_velocity + np.random.normal(scale = ang_vel_sig,size=3)
        translation_err    = translation      + np.random.normal(scale = translation_sig,size=3)
        height_err         = height_above_gr  + np.random.normal(scale = height_sig,size=1)
        flow_err           = true_flow        + np.random.normal(scale = flow_sig,size=(len(pos),2))
        position_err       = pos              + np.random.normal(scale = position_sig,size=(len(pos),2))
        normal_err         = normal_vector    + np.random.normal(scale = normal_sig,size=3)
        normal_err         = normal_vector    / np.linalg.norm(normal_vector)

        '''
        orient_err         = normal_sig*np.random.normal(scale=normal_sig)
        orient_err2       = normal_sig*np.random.normal(scale=normal_sig)
        orient_err3       = normal_sig*np.random.normal(scale=normal_sig)
        normal_err         = np.dot(np.dot(np.dot(np.array([[np.cos(orient_err2),0,np.sin(orient_err2)],[0,1,0],[-np.sin(orient_err2),0,np.cos(orient_err2)]]),np.array([[1,0,0],[0,np.cos(orient_err),-np.sin(orient_err)],[0,np.sin(orient_err),np.cos(orient_err)]])),np.array([[np.cos(orient_err3),0,np.sin(orient_err3)],[0,1,0],[-np.sin(orient_err3),0,np.cos(orient_err3)]])),normal_vector)
        '''

        v_obs[i],Res,singular  = solve_lgs(position_err,flow_err,height_err,normal_err,ang_vel_err,translation_err)
        part_error=np.zeros(len(pos))
        for j in range(len(pos)):
            xp    = np.array([pos[j,0],pos[j,1],1])
            dxp   = np.array([position_err[j,0],position_err[j,1],1])-xp
            ddotx = np.array([flow_err[j,0]-true_flow[j,0],flow_err[j,1]-true_flow[j,1],0])
            v_err = (height_err-height_above_gr)/height_above_gr*np.dot(normal_vector,xp)+np.dot((normal_err-normal_vector),xp)+np.dot(normal_vector,dxp)
            d_err = ddotx + np.cross(dxp,angular_velocity)+np.cross(xp,ang_vel_err-angular_velocity)+dxp
            part_error[j]=np.linalg.norm(np.cross(xp,(v_err)*linear_velocity+height_above_gr*d_err))/np.amin(singular)
        R[i]=np.sqrt(np.sum(part_error**2))+np.linalg.norm(angular_velocity)*translation_sig+ang_vel_sig*np.linalg.norm(translation)+ang_vel_sig*translation_sig
        feasible           = feasibility(position_err,linear_velocity,flow_err,ang_vel_err,translation_err,normal_err)
    return v_obs, feasible,R



def feas_simulation(linear_velocity,angular_velocity,height_above_gr,normal_vector,translation,pos,ang_vel_sig,translation_sig,height_sig,flow_sig,position_sig,normal_sigi,true_vel):
    v_obs         = np.zeros((iterations,3))
    R             = np.zeros(iterations)
    forward_para  = np.zeros((iterations,len(pos)))
    backward_para = np.zeros((iterations,len(pos)))
    forward_dist  = np.zeros((iterations,len(pos)))
    backward_dist = np.zeros((iterations,len(pos)))
    backward_res  = np.zeros((iterations,len(pos)))
    forward_res   = np.zeros((iterations,len(pos)))
    for i in range(iterations):
        ang_vel_err        = angular_velocity + np.random.normal(scale = ang_vel_sig,size=3)
        translation_err    = translation      + np.random.normal(scale = translation_sig,size=3)
        height_err         = height_above_gr  + np.random.normal(scale = height_sig,size=1)
        flow_err           = true_flow        + np.random.normal(scale = flow_sig,size=(200,2))
        position_err       = pos              + np.random.normal(scale = position_sig,size=(200,2))
        vel_err             = true_vel         + np.random.normal(scale = velocity_sig,size=3)

        orient_err         = normal_sig*np.random.normal(scale=normal_sig)
        orient_err2        = normal_sig*np.random.normal(scale=normal_sig)
        normal_err         = np.dot(np.dot(np.array([[np.cos(orient_err2),0,np.sin(orient_err2)],[0,1,0],[-np.sin(orient_err2),0,np.cos(orient_err2)]]),np.array([[1,0,0],[0,np.cos(orient_err),-np.sin(orient_err)],[0,np.sin(orient_err),np.cos(orient_err)]])),normal_vector)

        v_obs[i],R[i],singular      = solve_lgs(position_err,flow_err,height_err,normal_err,ang_vel_err,translation_err)
        backward_para[i],backward_dist[i]   = feasibility(position_err,v_obs[i],flow_err,ang_vel_err,translation_err,normal_err)
        forward_para[i],forward_dist[i]    = feasibility(position_err,vel_err,flow_err,ang_vel_err,translation_err,normal_err)
    
        x=position_err
        u=flow_err
        n=normal_err
        omega=ang_vel_err
        for j in range(len(x)):
            x_hat=np.array([[0,-1.0,x[j,1]],[1.0,0,-x[j,0]],[-x[j,1],x[j,0],0]]) 
            b_i = np.dot(x_hat,np.array([u[j,0],u[j,1],0])+1.00*np.dot(x_hat,omega))  #append 3rd dim for calculation (faster method ?)
            backward_res[i,j]=np.linalg.norm(np.dot(x_hat*np.dot(n,np.append(x[j],1)),v_obs[i])-b_i)
            forward_res[i,j]=np.linalg.norm(np.dot(x_hat*np.dot(n,np.append(x[j],1)),vel_err)-b_i)
    return np.mean(backward_para,axis=0),np.mean(backward_dist,axis=0),np.mean(forward_para,axis=0),np.mean(forward_dist,axis=0),np.mean(backward_res,axis=0),np.mean(forward_res,axis=0)



def feasibility(position,linear_velocity,flow,angular_velocity,translation,normal):
    parallelity = np.zeros(len(position))
    length       = np.zeros(len(position))
    for i in range(len(position)):
        fac1     = np.cross(np.array([position[i,0],position[i,1],1]),linear_velocity-np.cross(angular_velocity,translation)) 
        fac2      = np.cross(np.array([position[i,0],position[i,1],1]),np.array([flow[i,0],flow[i,1],0])-np.cross(angular_velocity,np.array([position[i,0],position[i,1],1])))
        fac1_norm = np.linalg.norm(fac1)
        fac2_norm = np.linalg.norm(fac2)

        parallelity[i] = np.dot(fac1,fac2)/(fac1_norm*fac2_norm)
        length[i]      = fac1_norm/fac2_norm*np.dot(normal,np.array([position[i,0],position[i,1],1]))

    return np.array([parallelity,length])
#load same points for each simulation to ensure continuity
data=np.loadtxt("points.txt")

def overlap(data1,data2):

    binedge   = np.histogram(np.hstack((data1,data2)),bins=100)[1]
    hist1     = np.histogram(data1,bins=binedge)
    hist2     = np.histogram(data2,bins=binedge)
    '''
    plt.hist([data1,data2],bins=binedge,label=["data1","data2"])
    plt.plot(hist1[1][:-1],hist1[0])
    plt.plot(hist2[1][:-1],hist2[0])
    plt.show()
    '''
    mini      = np.minimum(hist1[0],hist2[0])
    return np.sum(mini)


    '''
    mean1 = np.mean(data1)
    mean2 = np.mean(data2)

    std1  = np.std(data1)
    std2  = np.std(data1)

    if mean1 >= mean2:
        return mean2+std2-mean1+std1
    else:
        return mean1+std1-mean2+std2
    '''
#print(np.mean(data**2))
#True Movement of points
#--------------------------------------------
linear_velocity  = np.array([1,1,1])
angular_velocity = 1*np.array([1,1,1])
height_above_gr  = 1 
normal_vector    = np.array([0,0,1])
translation      = np.array([0.02,0,0.205])

#no rotation is assumed since R needs to be inverted first
true_flow     = generate_test_data(data,linear_velocity,angular_velocity,height_above_gr,normal_vector,translation)
#----------------------------------------------

#Measurement errors
#----------------------------------------------
ang_vel_sig     = 0.00071  #measured from std while idling
translation_sig = 0.005
height_sig      = 0.01
flow_sig        = 0.056*np.sqrt(2)*1.23  #CAREFULL STILL IN PIX!!!
position_sig    = 0.056*1.23             #CAREFULL STILL IN PIX!!!
normal_sig      = 0.00065   #measured from pixhawk using max(std(x,y,z))
velocity_sig    = 0.01      #placeholder value
#----------------------------------------------




iterations=100

'''
#Effect of flow errors:
#-------------------------------------------------------------------------------------------------------
k=100    #number of simulations
v_flow    =np.zeros((k,3))
v_flow_err=np.zeros((k,3))
data[:,0]=(data[:,0]-np.mean(data[:,0]))*1.27
data[:,1]=(data[:,1]-np.mean(data[:,1]))*0.93
normal_vector=np.array([0,0,1])
iterations=100
true_flow     = generate_test_data(data,linear_velocity,angular_velocity,height_above_gr,normal_vector,translation)
#----------------------------------------------
for i in range(k):
    print(i)
    v_obs,feasible,r= of_simulation(linear_velocity,angular_velocity,height_above_gr,normal_vector,translation,data,ang_vel_sig,translation_sig,height_sig,0.001*i,np.sqrt(2)/1000*i,normal_sig)
    v_flow[i,0]     = np.mean(v_obs[:,0])
    v_flow[i,1]     = np.mean(v_obs[:,1])
    v_flow[i,2]     = np.mean(v_obs[:,2])
    v_flow_err[i,0] = np.std(v_obs[:,0])
    v_flow_err[i,1] = np.std(v_obs[:,1])
    v_flow_err[i,2] = np.std(v_obs[:,2])

np.save("effect_of_flow_errors",np.append(v_flow,v_flow_err))
plt.errorbar(0.001*np.linspace(0,k,k),v_flow[:,0],v_flow_err[:,0],label="x",fmt="o")
plt.errorbar(0.001*np.linspace(0,k,k),v_flow[:,1],v_flow_err[:,1],label="y",fmt="o")
plt.errorbar(0.001*np.linspace(0,k,k),v_flow[:,2],v_flow_err[:,2],label="z",fmt="o")
plt.legend()
plt.xlabel("Fehler in Position [m] und Fluss [m/s]")
plt.ylabel("Geschwindigkeit [m/s]")
plt.title("Fehlerauswirkung von Tracking und Position bei v=[1,1,1]")
plt.show()
#----------------------------------------------------------------------------------------------------------


#Effect of distance_error:
#-------------------------------------------------------------------------------------------------------
k=100    #number of simulations
v_flow    =np.zeros((k,3))
v_flow_err=np.zeros((k,3))
data[:,0]=(data[:,0]-np.mean(data[:,0]))*1.27
data[:,1]=(data[:,1]-np.mean(data[:,1]))*0.93
normal_vector=np.array([0,0,1])
iterations=100
true_flow     = generate_test_data(data,linear_velocity,angular_velocity,height_above_gr,normal_vector,translation)
#----------------------------------------------
for i in range(k):
    print(i)
    v_obs,feasible,r= of_simulation(linear_velocity,angular_velocity,height_above_gr,normal_vector,translation,data,ang_vel_sig,translation_sig,0.001*i,flow_sig,position_sig,normal_sig)
    v_flow[i,0]     = np.mean(v_obs[:,0])
    v_flow[i,1]     = np.mean(v_obs[:,1])
    v_flow[i,2]     = np.mean(v_obs[:,2])
    v_flow_err[i,0] = np.std(v_obs[:,0])
    v_flow_err[i,1] = np.std(v_obs[:,1])
    v_flow_err[i,2] = np.std(v_obs[:,2])

np.save("effect_of_distance_error",np.append(v_flow,v_flow_err))
plt.errorbar(0.001*np.linspace(0,k,k),v_flow[:,0],v_flow_err[:,0],label="x",fmt="o")
plt.errorbar(0.001*np.linspace(0,k,k),v_flow[:,1],v_flow_err[:,1],label="y",fmt="o")
plt.errorbar(0.001*np.linspace(0,k,k),v_flow[:,2],v_flow_err[:,2],label="z",fmt="o")
plt.legend()
plt.xlabel("Fehler in Distanz zum Boden")
plt.ylabel("Geschwindigkeit")
plt.title("Fehlerauswirkung von Tracking und Position bei v=[1,1,1]")
plt.show()
#----------------------------------------------------------------------------------------------------------


#Effect of ang_vel_err:
#-------------------------------------------------------------------------------------------------------
k=100    #number of simulations
v_flow    =np.zeros((k,3))
v_flow_err=np.zeros((k,3))
data[:,0]=(data[:,0]-np.mean(data[:,0]))*1.27
data[:,1]=(data[:,1]-np.mean(data[:,1]))*0.93
normal_vector=np.array([0,0,1])
iterations=100
true_flow     = generate_test_data(data,linear_velocity,angular_velocity,height_above_gr,normal_vector,translation)
#----------------------------------------------
for i in range(k):
    print(i)
    v_obs,feasible,r= of_simulation(linear_velocity,angular_velocity,height_above_gr,normal_vector,translation,data,0.001*i,translation_sig,height_sig,flow_sig,position_sig,normal_sig)
    v_flow[i,0]     = np.mean(v_obs[:,0])
    v_flow[i,1]     = np.mean(v_obs[:,1])
    v_flow[i,2]     = np.mean(v_obs[:,2])
    v_flow_err[i,0] = np.std(v_obs[:,0])
    v_flow_err[i,1] = np.std(v_obs[:,1])
    v_flow_err[i,2] = np.std(v_obs[:,2])




np.save("effect_o_ang_vel_error",np.append(v_flow,v_flow_err))
plt.errorbar(0.001*np.linspace(0,k,k),v_flow[:,0],v_flow_err[:,0],label="x",fmt="o")
plt.errorbar(0.001*np.linspace(0,k,k),v_flow[:,1],v_flow_err[:,1],label="y",fmt="o")
plt.errorbar(0.001*np.linspace(0,k,k),v_flow[:,2],v_flow_err[:,2],label="z",fmt="o")
plt.legend()
plt.xlabel("Fehler in Drehgeschwindigkeit")
plt.ylabel("Geschwindigkeit")
plt.title("Fehlerauswirkung von Tracking und Position bei v=[1,1,1]")
plt.show()
#----------------------------------------------------------------------------------------------------------


#Effect of normal_err:
#-------------------------------------------------------------------------------------------------------
k=100    #number of simulations
v_flow    =np.zeros((k,3))
v_flow_err=np.zeros((k,3))
data[:,0]=(data[:,0]-np.mean(data[:,0]))*1.27
data[:,1]=(data[:,1]-np.mean(data[:,1]))*0.93
normal_vector=np.array([0,0,1])
iterations=10
true_flow     = generate_test_data(data,linear_velocity,angular_velocity,height_above_gr,normal_vector,translation)
#----------------------------------------------
for i in range(k):
    print(i)
    v_obs,feasible,r= of_simulation(linear_velocity,angular_velocity,height_above_gr,normal_vector,translation,data,ang_vel_sig,translation_sig,height_sig,flow_sig,position_sig,0.001*i)
    v_flow[i,0]     = np.mean(v_obs[:,0])
    v_flow[i,1]     = np.mean(v_obs[:,1])
    v_flow[i,2]     = np.mean(v_obs[:,2])
    v_flow_err[i,0] = np.std(v_obs[:,0])
    v_flow_err[i,1] = np.std(v_obs[:,1])
    v_flow_err[i,2] = np.std(v_obs[:,2])

np.save("effect_of_normal_error",np.append(v_flow,v_flow_err))
plt.errorbar(0.001*np.linspace(0,k,k),v_flow[:,0],v_flow_err[:,0],label="x",fmt="o")
plt.errorbar(0.001*np.linspace(0,k,k),v_flow[:,1],v_flow_err[:,1],label="y",fmt="o")
plt.errorbar(0.001*np.linspace(0,k,k),v_flow[:,2],v_flow_err[:,2],label="z",fmt="o")
plt.legend()
plt.xlabel("Fehler in Normalvektor")
plt.ylabel("Geschwindigkeit")
plt.title("Fehlerauswirkung von Tracking und Position bei v=[1,1,1]")
plt.show()
#----------------------------------------------------------------------------------------------------------



#Effect of translation_err:
#-------------------------------------------------------------------------------------------------------
k=100    #number of simulations
v_flow    =np.zeros((k,3))
v_flow_err=np.zeros((k,3))
data[:,0]=(data[:,0]-np.mean(data[:,0]))*1.27
data[:,1]=(data[:,1]-np.mean(data[:,1]))*0.93
normal_vector=np.array([0,0,1])
iterations=100
true_flow     = generate_test_data(data,linear_velocity,angular_velocity,height_above_gr,normal_vector,translation)
#----------------------------------------------
for i in range(k):
    print(i)
    v_obs,feasible,r= of_simulation(linear_velocity,angular_velocity,height_above_gr,normal_vector,translation,data,ang_vel_sig,0.001*i,height_sig,flow_sig,position_sig,normal_sig)
    v_flow[i,0]     = np.mean(v_obs[:,0])
    v_flow[i,1]     = np.mean(v_obs[:,1])
    v_flow[i,2]     = np.mean(v_obs[:,2])
    v_flow_err[i,0] = np.std(v_obs[:,0])
    v_flow_err[i,1] = np.std(v_obs[:,1])
    v_flow_err[i,2] = np.std(v_obs[:,2])

np.save("effect_of_translation_error",np.append(v_flow,v_flow_err))
plt.errorbar(0.001*np.linspace(0,k,k),v_flow[:,0],v_flow_err[:,0],label="x",fmt="o")
plt.errorbar(0.001*np.linspace(0,k,k),v_flow[:,1],v_flow_err[:,1],label="y",fmt="o")
plt.errorbar(0.001*np.linspace(0,k,k),v_flow[:,2],v_flow_err[:,2],label="z",fmt="o")
plt.legend()
plt.xlabel("Fehler in Position und Fluss")
plt.ylabel("Geschwindigkeit")
plt.title("Fehlerauswirkung von Tracking und Position bei v=[1,1,1]")
plt.show()
#----------------------------------------------------------------------------------------------------------






#Effect of orientation
#------------------------------------------------------------------------------------------------------------

data[:,0]=(data[:,0]-np.mean(data[:,0]))*1.27
data[:,1]=(data[:,1]-np.mean(data[:,1]))*0.93
k=100    #number of simulations
v_flow    =np.zeros((k,3))
v_flow_err=np.zeros((k,3))
iterations=10
for i in range(k):
    print(i)
    normal_vector       = np.array([np.sin(np.pi*i/k),0,np.cos(np.pi*i/k)])
    true_flow           = generate_test_data(data,linear_velocity,angular_velocity,height_above_gr,normal_vector,translation)
    v_obs,feasible,R_i  =           of_simulation(linear_velocity,angular_velocity,height_above_gr,normal_vector,translation,
                                                  data,ang_vel_sig,translation_sig,height_sig,flow_sig,position_sig,normal_sig)
    v_flow[i,0]     = np.mean(v_obs[:,0])
    v_flow[i,1]     = np.mean(v_obs[:,1])
    v_flow[i,2]     = np.mean(v_obs[:,2])
    v_flow_err[i,0] = np.std(v_obs[:,0])
    v_flow_err[i,1] = np.std(v_obs[:,1])
    v_flow_err[i,2] = np.std(v_obs[:,2])


np.save("effect_of_orientation",np.append(v_flow,v_flow_err))
#plt.scatter(180.0/k*np.linspace(0,k,k),v_flow_err[:,0],label="x")
#plt.scatter(180.0/k*np.linspace(0,k,k),v_flow_err[:,1],label="y")
#plt.scatter(180.0/k*np.linspace(0,k,k),v_flow_err[:,2],label="z")
#plt.xlabel("Winkel zum Boden ")
#plt.ylabel("Standartabweichung der  Geschwindigkeit")
#plt.title("Geschwindigkeitsfehler durch Orientierung zum Boden")
#plt.legend()
#plt.show()
plt.errorbar(180*np.linspace(0,k,k)/k,v_flow[:,0],v_flow_err[:,0],label="x",fmt="o")
plt.errorbar(180*np.linspace(0,k,k)/k,v_flow[:,1],v_flow_err[:,1],label="y",fmt="o")
plt.errorbar(180*np.linspace(0,k,k)/k,v_flow[:,2],v_flow_err[:,2],label="z",fmt="o")
plt.legend()
plt.xlabel("Winkel zum Boden ")
plt.ylabel("Geschwindigkeit")
plt.title("Fehlerauswirkung von Tracking und Position bei v=[1,1,1]")
plt.show()
#-------------------------------------------------------------------------------------------------------------


'''
#Effect of Height 
#------------------------------------------------------------------------------------------------------------
'''
data[:,0]=(data[:,0]-np.mean(data[:,0]))*1.27#*10
data[:,1]=(data[:,1]-np.mean(data[:,1]))*0.93#*10
k=100    #number of simulations
v_flow    =np.zeros((k,3))
v_flow_err=np.zeros((k,3))
print("start")
normal_vector=np.array([0,0,1])
#angular_velocity=np.array([0,0,1])
#linear_velocity=np.array([1,1,0])
iterations=100
for i in range(k):
    print(i)
    true_flow     = generate_test_data(data,linear_velocity,angular_velocity,0.2+np.linspace(0.2,7.65,k)[i],normal_vector,translation)
    v_obs,feasible,r  = of_simulation(linear_velocity,angular_velocity      ,0.2+np.linspace(0.2,7.65,k)[i],normal_vector,translation,
                        data,ang_vel_sig,translation_sig,height_sig,flow_sig,position_sig,normal_sig)
    v_flow[i,0]     = np.mean(v_obs[:,0])
    v_flow[i,1]     = np.mean(v_obs[:,1])
    v_flow[i,2]     = np.mean(v_obs[:,2])
    v_flow_err[i,0] = np.std(v_obs[:,0])
    v_flow_err[i,1] = np.std(v_obs[:,1])
    v_flow_err[i,2] = np.std(v_obs[:,2])

np.save("effect_of_height",np.append(v_flow,v_flow_err))
print(np.polyfit(0.2+np.linspace(0.2,7.65,k),v_flow[:,0],deg=1))
print(np.polyfit(0.2+np.linspace(0.2,7.65,k),v_flow[:,1],deg=1))
print(np.polyfit(0.2+np.linspace(0.2,7.65,k),v_flow[:,2],deg=1))
plt.errorbar(0.2+np.linspace(0.2,7.65,k),v_flow[:,0],v_flow_err[:,0],label="x-Richtung",fmt="o")
plt.errorbar(0.2+np.linspace(0.2,7.65,k),v_flow[:,1],v_flow_err[:,1],label="y-Richtung",fmt="o")
plt.errorbar(0.2+np.linspace(0.2,7.65,k),v_flow[:,2],v_flow_err[:,2],label="z-Richtung",fmt="o")
plt.tick_params(labelsize=20)
plt.legend(prop={'size':20})
plt.xlabel("Distanz zum  Boden [m]",fontsize=24)
plt.ylabel("Geschwindigkeit [m/s]",fontsize=24)
plt.title("Fehlerauswirkung von Abstand bei v=[1,1,1]")
plt.show()
'''


'''
#Effekt of point position
#------------------------------------------------------------------------------------------------------------
data[:,0]=data[:,0]-np.mean(data[:,0])*1.27
data[:,1]=data[:,1]-np.mean(data[:,1])*0.93
k=100    #number of simulations
v_flow    =np.zeros((k,3))
v_flow_err=np.zeros((k,3))
iterations=100
for i in range(k):
    print(i)
    true_flow     = generate_test_data(data+np.ones_like(data)*i/100,linear_velocity,angular_velocity,height_above_gr,normal_vector,translation)
    v_obs,feasible,r  = of_simulation(linear_velocity,angular_velocity,height_above_gr,normal_vector,translation,
                        data+np.ones_like(data)*i/100,ang_vel_sig,translation_sig,height_sig,flow_sig,position_sig,normal_sig)
    v_flow[i,0]     = np.mean(v_obs[:,0])
    v_flow[i,1]     = np.mean(v_obs[:,1])
    v_flow[i,2]     = np.mean(v_obs[:,2])
    v_flow_err[i,0] = np.std(v_obs[:,0])
    v_flow_err[i,1] = np.std(v_obs[:,1])
    v_flow_err[i,2] = np.std(v_obs[:,2])
    print(np.mean(data+np.ones_like(data)*i/100))

np.save("effect_of_point_position",np.append(v_flow,v_flow_err))
plt.errorbar(np.linspace(0,k,k)/k,v_flow[:,0],v_flow_err[:,0],label="x",fmt="o")
plt.errorbar(np.linspace(0,k,k)/k,v_flow[:,1],v_flow_err[:,1],label="y",fmt="o")
plt.errorbar(np.linspace(0,k,k)/k,v_flow[:,2],v_flow_err[:,2],label="z",fmt="o")
plt.legend()
plt.xlabel("Verschiebung der Punkte in m")
plt.ylabel("Geschwindigkeit")
plt.title("Fehlerauswirkung von Punktposition ")
plt.show()

# Time analysis
data[:,0]=data[:,0]-np.mean(data[:,0])
data[:,1]=data[:,1]-np.mean(data[:,1])
data=data*10
k=100    #number of simulations
v_flow    =np.zeros((k,3))
v_flow_err=np.zeros((k,3))
for i in range(k):
    print(i)
    true_flow     = generate_test_data(data,linear_velocity,angular_velocity,height_above_gr,normal_vector,translation)
    v_obs,feasible,r  = of_simulation(linear_velocity,angular_velocity,height_above_gr,normal_vector,translation,
                        data,ang_vel_sig,translation_sig,height_sig,flow_sig,position_sig,normal_sig)
    v_flow[i,0]     = np.mean(v_obs[:,0])
    v_flow[i,1]     = np.mean(v_obs[:,1])
    v_flow[i,2]     = np.mean(v_obs[:,2])
    v_flow_err[i,0] = np.std(v_obs[:,0])
    v_flow_err[i,1] = np.std(v_obs[:,1])
    v_flow_err[i,2] = np.std(v_obs[:,2])
    
    #change dynamic parameters
    data=data+ generate_test_data(data,linear_velocity,np.array([0,0,0]),height_above_gr,normal_vector,translation)
    print(data)
    height_above_gr=height_above_gr+np.dot(linear_velocity,normal_vector)
plt.errorbar(np.linspace(0,k,k),v_flow[:,0],v_flow_err[:,0],label="x",fmt="o")
plt.errorbar(np.linspace(0,k,k),v_flow[:,1],v_flow_err[:,1],label="y",fmt="o")
plt.errorbar(np.linspace(0,k,k),v_flow[:,2],v_flow_err[:,2],label="z",fmt="o")
plt.legend()
plt.xlabel("Zeitschritt")
plt.ylabel("Geschwindigkeit")
plt.title("Fehlerauswirkung bei Senkrechtem Flug von einem Meter aus")
plt.show()

'''
#sim_err vs numerical error
#-------------------------------------------------------------------------------------------------------------------------------
'''
data[:,0]=(data[:,0]-np.mean(data[:,0]))*1.27
data[:,1]=(data[:,1]-np.mean(data[:,1]))*0.93
k=100    #number of simulations
v_flow    =np.zeros((k,3))
v_flow_err=np.zeros((k,3))
numm_err   =np.zeros(k)
print(np.mean(data[:,0]))
print(np.mean(data[:,1]))
normal_vector=np.array([0,0,1])
iterations=10
print("start")
for i in range(k):
    print(i)
    true_flow     = generate_test_data(data,linear_velocity,angular_velocity,0.2+np.linspace(0.2,7.65,k)[i],normal_vector,translation)
    v_obs,feasible,r  = of_simulation(linear_velocity,angular_velocity      ,0.2+np.linspace(0.2,7.65,k)[i],normal_vector,translation,
                        data,ang_vel_sig,translation_sig,height_sig,flow_sig,position_sig,normal_sig)
    v_flow[i,0]     = np.mean(v_obs[:,0])
    v_flow[i,1]     = np.mean(v_obs[:,1])
    v_flow[i,2]     = np.mean(v_obs[:,2])
    numm_err        = np.mean(r)
    v_flow_err[i,0] = np.std(v_obs[:,0])
    v_flow_err[i,1] = np.std(v_obs[:,1])
    v_flow_err[i,2] = np.std(v_obs[:,2])


np.save("sim_err_vs_num2",np.append(v_flow,numm_err))
print(np.polyfit(0.2+np.linspace(0.2,7.65,k),v_flow[:,0],deg=1))
print(np.polyfit(0.2+np.linspace(0.2,7.65,k),v_flow[:,1],deg=1))
print(np.polyfit(0.2+np.linspace(0.2,7.65,k),v_flow[:,2],deg=1))
plt.errorbar(0.2+np.linspace(0.2,7.65,k),v_flow[:,0],v_flow_err[:,0],label="x",fmt="o")
plt.errorbar(0.2+np.linspace(0.2,7.65,k),v_flow[:,1],v_flow_err[:,1],label="y",fmt="o")
plt.errorbar(0.2+np.linspace(0.2,7.65,k),v_flow[:,2],v_flow_err[:,2],label="z",fmt="o")
plt.errorbar(0.2+np.linspace(0.2,7.65,k),10000*np.ones(k),numm_err,label="numerical",fmt="x")
plt.legend()
plt.xlabel("Abstand von Boden in m")
plt.ylabel("Geschwindigkeit")
plt.title("Fehlerauswirkung von Abstand bei v=[1,1,1]")
plt.show()




#number of Point simulation
#-------------------------------------------------------------------------------------------------------------------------------

data[:,0]=data[:,0]-np.mean(data[:,0])*1.27
data[:,1]=data[:,1]-np.mean(data[:,1])*0.93
k=99    #number of simulations
v_flow    =np.zeros((k,3))
v_flow_err=np.zeros((k,3))
numm_err   =np.zeros(k)
iterations=10
#normal_vector=np.array([0,0,1])
Time = np.zeros(k)
for i in range(k):
    print(i)
    true_flow     = generate_test_data(data[:2*i+2],linear_velocity,angular_velocity,height_above_gr,normal_vector,translation)
    start         = time.time()
    v_obs,feasible,r  = of_simulation(linear_velocity,angular_velocity,height_above_gr,normal_vector,translation,
            data[:2*i+2],ang_vel_sig,translation_sig,height_sig,flow_sig,position_sig,normal_sig)
    stop          = time.time()
    Time[i]       = (stop-start)/100
    v_flow[i,0]     = np.mean(v_obs[:,0])
    v_flow[i,1]     = np.mean(v_obs[:,1])
    v_flow[i,2]     = np.mean(v_obs[:,2])
    numm_err[i]     = np.mean(r)
    v_flow_err[i,0] = np.std(v_obs[:,0])
    v_flow_err[i,1] = np.std(v_obs[:,1])
    v_flow_err[i,2] = np.std(v_obs[:,2])

np.save("number_of_point",np.append(v_flow,v_flow_err))
np.save("sim_err_vs_num2",numm_err)
#fit time scaling 
print(np.polyfit(2+np.linspace(0,k,k),Time,deg=1))
plt.errorbar(2*np.linspace(0,k,k)+2,np.ones(k),numm_err,label="numerical",fmt="x")
plt.errorbar(2*np.linspace(0,k,k)+2,v_flow[:,0],v_flow_err[:,0],label="x",fmt="o")
plt.errorbar(2*np.linspace(0,k,k)+2,v_flow[:,1],v_flow_err[:,1],label="y",fmt="o")
plt.errorbar(2*np.linspace(0,k,k)+2,v_flow[:,2],v_flow_err[:,2],label="z",fmt="o")
plt.legend()
plt.xlabel("Features")
plt.ylabel("Geschwindigkeit")
plt.title("Geschwindgikeits in Abhaengigkeit von Feature Zahl")
plt.show()


plt.scatter(2*np.linspace(0,k,k)+2,Time)
plt.xlabel("Features")
plt.ylabel("Zeit Pro Evaluation [s]")
plt.title("Zeitverhalten bei zunehmender Featurezahl")
plt.show()
'''







#Sorting Simulations-----------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------





#vergleich backward vs forward
#------------------------------------------------------------------------------------------------------------------------------------------
'''
data[:,0]=(data[:,0]-np.mean(data[:,0]))*1.27
data[:,1]=(data[:,1]-np.mean(data[:,1]))*0.93

height_above_gr=3
#normal_vector=np.array([0,0,1])
k=100    #number of simulations
v_flow    =np.zeros((k,3))
v_flow_err=np.zeros((k,3))
#linear_velocity=linear_velocity*10

first_plane        = generate_test_data(data[:int(len(data)/3)],linear_velocity,angular_velocity,height_above_gr,normal_vector,translation)
second_plane       = generate_test_data(data[int(len(data)/3):2*int(len(data)/3)],linear_velocity,angular_velocity,height_above_gr-1,normal_vector,translation)
third_plane       = generate_test_data(data[2*int(len(data)/3):],linear_velocity,angular_velocity,height_above_gr-2,normal_vector,translation)
true_flow=np.append(first_plane,np.append(second_plane,third_plane,axis=0),axis=0)
backward_para,backward_dist, forward_para,forward_dist,backward_res,forward_res= feas_simulation(linear_velocity,angular_velocity,height_above_gr,normal_vector,translation,
                                                            data,ang_vel_sig,translation_sig,height_sig,flow_sig,position_sig,normal_sig,linear_velocity)

sorted_distance = np.sort(forward_dist)
distance_diff   = [j-i for i,j in zip(sorted_distance[:-1],sorted_distance[1:])]

plt.plot(distance_diff,label=u"Höhendifferenz")
plt.xlabel("Feature",fontsize=24)
plt.ylabel(u"Höhe [m]",fontsize=24)
plt.scatter(np.linspace(0,len(sorted_distance),len(sorted_distance)),sorted_distance, label=u"sortierte Höhen")
plt.tick_params(labelsize=20)
plt.legend(prop={'size': 20})
plt.show()

hist=plt.figure(1)
hist.text(0.52, 0.02, 'Metrik', ha='center',fontsize=28)
hist.text(0.02, 0.5, 'Vorkommen', va='center',rotation='vertical',fontsize=28)

plot1=hist.add_subplot(321)
plot1.set_title(u"Parallelität (RW)",size=24)
plot1.hist([backward_para[:int(len(data)/3)],backward_para[int(len(data)/3):2*int(len(data)/3)],backward_para[2*int(len(data)/3):]],label=['3 Meter', '2 Meter','1 Meter'],bins=50)
plt.tick_params(labelsize=20)

plot2=hist.add_subplot(322)
plot2.set_title(u"Parallelität (VW)",size=24)
plot2.hist([forward_para[:int(len(data)/3)],forward_para[int(len(data)/3):2*int(len(data)/3)],forward_para[2*int(len(data)/3):]],label=["3 Meter","2 Meter","1 Meter"],bins=50)
plt.tick_params(labelsize=20)

plot3=hist.add_subplot(323)
plot3.set_title(u"Distanz (RW)",size=24)
plot3.hist([backward_dist[:int(len(data)/3)],backward_dist[int(len(data)/3):2*int(len(data)/3)],backward_dist[2*int(len(data)/3):]],label=["3 Meter","2 Meter","3 Meter"],bins=50)
plt.tick_params(labelsize=20)

plot4=hist.add_subplot(324)
plot4.set_title(u"Distanz (VW)",size=24)
plot4.hist([forward_dist[:int(len(data)/3)],forward_dist[int(len(data)/3):2*int((len(data)/3))],forward_dist[2*int(len(data)/3):]],label=["3 Meter","2 Meter","1 Meter"],bins=50)
plt.tick_params(labelsize=20)

plot5=hist.add_subplot(325)
plot5.set_title(u"|Residuum| (RW)",size=24)
plot5.hist([backward_res[:int(len(data)/3)],backward_res[int(len(data)/3):2*int(len(data)/3)],backward_res[2*int(len(data)/3):]],label=["3 Meter","2 Meter","1 Meter"],bins=50)
plt.tick_params(labelsize=20)

plot6=hist.add_subplot(326)
plot6.set_title(u"|Residuum| (VW)",size=24)
plot6.hist([forward_res[:int(len(data)/3)],forward_res[int(len(data)/3):2*int(len(data)/3)],forward_res[2*int(len(data)/3):]],label=["3 Meter","2 Meter","1 Meter"],bins=50)
plt.tick_params(labelsize=20)
plot6.legend(prop={'size': 20})

plt.tight_layout()
plt.show()
'''








# aussortieren bewegter Punkte.
#---------------------------------------------------------------------------------------

'''
data[:,0]=data[:,0]-np.mean(data[:,0])
data[:,1]=data[:,1]-np.mean(data[:,1])
data=data
k=100    #number of simulations
v_flow    =np.zeros((k,3))
v_flow_err=np.zeros((k,3))
linear_velocity=linear_velocity*10

first_plane        = generate_test_data(data[:int(len(data)/2)],linear_velocity,angular_velocity,height_above_gr,normal_vector,translation)
second_plane       = generate_test_data(data[int(len(data)/2):],linear_velocity,angular_velocity,height_above_gr,normal_vector,translation)
print(second_plane.shape)
for i in range(len(second_plane)):
    minang=10/360*2*np.pi
    angle= np.random.uniform(minang,2*np.pi-minang)
    second_plane[i,:]=np.dot(np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]).reshape(2,2),second_plane[i,:])
true_flow=np.append(first_plane,second_plane,axis=0)
backward_para,backward_dist, forward_para,forward_dist,backward_res,forward_res= feas_simulation(linear_velocity,angular_velocity,height_above_gr,normal_vector,translation,
                                                            data,ang_vel_sig,translation_sig,height_sig,flow_sig,position_sig,normal_sig,linear_velocity)
hist=plt.figure(1)
hist.text(0.52, 0.02, 'Metrik', ha='center',fontsize=28)
hist.text(0.02, 0.5, 'Vorkommen', va='center',rotation='vertical',fontsize=28)

plot1=hist.add_subplot(321)
plot1.set_title(u"Parallelität (RW)",size=24)
plot1.hist([backward_para[:int(len(data)/2)],backward_para[int(len(data)/2):]],label=["unbewegt","bewegt","1 Meter"],bins=50)
plt.tick_params(labelsize=20)

plot2=hist.add_subplot(322)
plot2.set_title(u"Parallelität (VW)",size=24)
plot2.hist([forward_para[:int(len(data)/2)],forward_para[int(len(data)/2):]],label=["unbewegt","bewegt","1 Meter"],bins=50)
plt.tick_params(labelsize=20)

plot3=hist.add_subplot(323)
plot3.set_title(u"Distanz (RW)",size=24)
plot3.hist([backward_dist[:int(len(data)/2)],backward_dist[int(len(data)/2):]],label=["unbewegt","bewegt","1 Meter"],bins=50)
plt.tick_params(labelsize=20)

plot4=hist.add_subplot(324)
plot4.set_title(u"Distanz (VW)",size=24)
plot4.hist([forward_dist[:int(len(data)/2)],forward_dist[int(len(data)/2):]],label=["unbewegt","bewegt","1 Meter"],bins=50)
plt.tick_params(labelsize=20)

plot5=hist.add_subplot(325)
plot5.set_title(u"|Residuum| (RW)",size=24)
plot5.hist([backward_res[:int(len(data)/2)],backward_res[int(len(data)/2):]],label=["unbewegt","bewegt","1 Meter"],bins=50)
plt.tick_params(labelsize=20)

plot6=hist.add_subplot(326)
plot6.set_title(u"|Residuum| (VW)",size=24)
plot6.hist([forward_res[:int(len(data)/2)],forward_res[int(len(data)/2):]],label=["unbewegt","bewegt","1 Meter"],bins=50)
plt.tick_params(labelsize=20)
plot6.legend(prop={'size': 15})

plt.tight_layout()
plt.show()
'''

#aussortieren von bewegt, und paralleler ebene
#------------------------------------------------------------------------------------------------------------------------------------------

data[:,0]=data[:,0]-np.mean(data[:,0])
data[:,1]=data[:,1]-np.mean(data[:,1])

height_above_gr=1
#normal_vector=np.array([0,0,1])
k=100    #number of simulations
v_flow    =np.zeros((k,3))
v_flow_err=np.zeros((k,3))

height_above_gr= 2
linear_velocity=linear_velocity*2.9*height_above_gr

first_plane        = generate_test_data(data[:int(len(data)/5)],linear_velocity,angular_velocity,height_above_gr,normal_vector,translation)
second_plane       = generate_test_data(data[int(len(data)/5):2*int(len(data)/3)],linear_velocity,angular_velocity,height_above_gr-1,normal_vector,translation)
for i in range(len(second_plane)):
    minang=10/360*2*np.pi
    angle= np.random.uniform(minang,2*np.pi-minang)
    second_plane[i,:]=np.dot(np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]).reshape(2,2),second_plane[i,:])
third_plane       = generate_test_data(data[2*int(len(data)/3):],linear_velocity,angular_velocity,height_above_gr-1,normal_vector,translation)
true_flow=np.append(first_plane,np.append(second_plane,third_plane,axis=0),axis=0)
backward_para,backward_dist, forward_para,forward_dist,backward_res,forward_res= feas_simulation(linear_velocity,angular_velocity,height_above_gr,normal_vector,translation,
                                                            data,ang_vel_sig,translation_sig,height_sig,flow_sig,position_sig,normal_sig,linear_velocity)
hist=plt.figure(1)
hist.text(0.52, 0.02, 'Metrik', ha='center',fontsize=28)
hist.text(0.02, 0.5, 'Vorkommen', va='center',rotation='vertical',fontsize=28)
print("aussortierte features:",np.sum([forward_para[int(len(data)/3):2*int(len(data)/3)] > 0.88])/float(int(len(data)/3)))
print("aussortierte features:",np.sum([backward_para[int(len(data)/3):2*int(len(data)/3)] > 0.88])/float(int(len(data)/3)))
plot1=hist.add_subplot(321)
plot1.set_title(u"Parallelität (RW)",size=24)
plot1.hist([backward_para[:int(len(data)/5)],backward_para[int(len(data)/5):2*int(len(data)/3)],backward_para[2*int(len(data)/3):]],label=["unbwegt 2 Meter", "bewegt 1 Meter","unbewegt 1 Meter"],bins=50)
plt.tick_params(labelsize=20)

plot2=hist.add_subplot(322)
plot2.set_title(u"Parallelität (VW)",size=24)
plot2.hist([forward_para[:int(len(data)/5)],forward_para[int(len(data)/5):2*int(len(data)/3)],forward_para[2*int(len(data)/3):]],label=["unbwegt 2 Meter", "bewegt 1 Meter","unbewegt 1 Meter"],bins=50)
plt.tick_params(labelsize=20)

plot3=hist.add_subplot(323)
plot3.set_title(u"Distanz (RW)",size=24)
plot3.hist([backward_dist[:int(len(data)/5)],backward_dist[int(len(data)/5):2*int(len(data)/3)],backward_dist[2*int(len(data)/3):]],label=["unbwegt 2 Meter", "bewegt 1 Meter","unbewegt 1 Meter"],bins=50)
plt.tick_params(labelsize=20)

plot4=hist.add_subplot(324)
plot4.set_title(u"Distanz (VW)",size=24)
plot4.hist([forward_dist[:int(len(data)/5)],forward_dist[int(len(data)/5):2*int((len(data)/3))],forward_dist[2*int(len(data)/3):]],label=["unbwegt 2 Meter", "bewegt 1 Meter","unbewegt 1 Meter"],bins=50)
plt.tick_params(labelsize=20)

plot5=hist.add_subplot(325)
plot5.set_title(u"|Residuum| (RW)",size=24)
plot5.hist([backward_res[:int(len(data)/5)],backward_res[int(len(data)/5):2*int(len(data)/3)],backward_res[2*int(len(data)/3):]],label=["unbwegt 2 Meter", "bewegt 1 Meter","unbewegt 1 Meter"],bins=50)
plt.tick_params(labelsize=20)

plot6=hist.add_subplot(326)
plot6.set_title(u"|Residuum| (VW)",size=24)
plot6.hist([forward_res[:int(len(data)/5)],forward_res[int(len(data)/5):2*int(len(data)/3)],forward_res[2*int(len(data)/3):]],label=["unbewegt 2 Meter", "bewegt 1 Meter","unbewegt 1 Meter"],bins=50)
plt.tick_params(labelsize=20)
plot6.legend(prop={'size':20})

plt.tight_layout()
plt.show()






#dynamik sorting
'''
data[:,0]=data[:,0]-np.mean(data[:,0])
data[:,1]=data[:,1]-np.mean(data[:,1])

height_above_gr=1
normal_vector=np.array([0,0,1])
k=100    #number of simulations
v_flow    =np.zeros((k,3))
v_flow_err=np.zeros((k,3))

mov_overlap_para_b   = np.zeros(k)  
mov_overlap_para_v   = np.zeros(k)
mov_overlap_dist_b   = np.zeros(k)
mov_overlap_dist_v   = np.zeros(k)
mov_overlap_res_b    = np.zeros(k)
mov_overlap_res_v    = np.zeros(k)
plane_overlap_para_b = np.zeros(k)
plane_overlap_para_v = np.zeros(k)
plane_overlap_dist_b = np.zeros(k)
plane_overlap_dist_v = np.zeros(k)
plane_overlap_res_b  = np.zeros(k)
plane_overlap_res_v  = np.zeros(k)

iterations = 100

for i in range(k):
    print(i,"%")
    linear_velocity=linear_velocity*i*0.05
    first_plane        = generate_test_data(data[:int(len(data)/3)],linear_velocity,angular_velocity,height_above_gr,normal_vector,translation)
    second_plane       = generate_test_data(data[int(len(data)/3):2*int(len(data)/3)],linear_velocity,angular_velocity,height_above_gr,normal_vector,translation)
    for j in range(len(second_plane)):
        minang=1
        angle= np.random.uniform(minang,2*np.pi-minang)
        second_plane[j,:]=np.dot(np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]).reshape(2,2),second_plane[j,:])

    third_plane       = generate_test_data(data[2*int(len(data)/3):],linear_velocity,angular_velocity,height_above_gr+1,normal_vector,translation)
    true_flow=np.append(first_plane,np.append(second_plane,third_plane,axis=0),axis=0)

    backward_para,backward_dist, forward_para,forward_dist,backward_res,forward_res= feas_simulation(linear_velocity,angular_velocity,height_above_gr,normal_vector,translation,
                                                            data,ang_vel_sig,translation_sig,height_sig,flow_sig,position_sig,normal_sig,linear_velocity)

    mov_overlap_para_b[i] =overlap(backward_para[:int(len(data)/3)],backward_para[int(len(data)/3):2*int(len(data)/3)])
    mov_overlap_para_v[i] =overlap(forward_para[:int(len(data)/3)]  ,forward_para[int(len(data)/3):2*int(len(data)/3)])
    mov_overlap_dist_b[i] =overlap(backward_dist[:int(len(data)/3)],backward_dist[int(len(data)/3):2*int(len(data)/3)])
    mov_overlap_dist_v[i] =overlap(forward_dist[:int(len(data)/3)]  ,forward_dist[int(len(data)/3):2*int(len(data)/3)])
    mov_overlap_res_b[i]  =overlap(backward_res[:int(len(data)/3)]  ,backward_res[int(len(data)/3):2*int(len(data)/3)])
    mov_overlap_res_v[i]  =overlap(forward_res[:int(len(data)/3)]    ,forward_res[int(len(data)/3):2*int(len(data)/3)])


    plane_overlap_para_b[i] =overlap(backward_para[:int(len(data)/3)],backward_para[2*int(len(data)/3):])
    plane_overlap_para_v[i] =overlap(forward_para[:int(len(data)/3)]  ,forward_para[2*int(len(data)/3):])
    plane_overlap_dist_b[i] =overlap(backward_dist[:int(len(data)/3)],backward_dist[2*int(len(data)/3):])
    plane_overlap_dist_v[i] =overlap(forward_dist[:int(len(data)/3)]  ,forward_dist[2*int(len(data)/3):])
    plane_overlap_res_b[i]  =overlap(backward_res[:int(len(data)/3)]  ,backward_res[2*int(len(data)/3):])
    plane_overlap_res_v[i]  =overlap(forward_res[:int(len(data)/3)]    ,forward_res[2*int(len(data)/3):])


plt.plot(np.linspace(0,5,100),mov_overlap_para_b,label="    mov_overlap_para_b")
plt.plot(np.linspace(0,5,100),mov_overlap_para_v,label="    mov_overlap_para_v")
plt.plot(np.linspace(0,5,100),mov_overlap_dist_b,label="    mov_overlap_dist_b")
plt.plot(np.linspace(0,5,100),mov_overlap_dist_v,label="    mov_overlap_dist_v")
plt.plot(np.linspace(0,5,100),mov_overlap_res_b,label="    mov_overlap_res_b")
plt.plot(np.linspace(0,5,100),mov_overlap_res_v,label="    mov_overlap_res_v")
plt.legend()
plt.show()

plt.plot(np.linspace(0,5,100),plane_overlap_para_b,label="  plane_overlap_para_b")
plt.plot(np.linspace(0,5,100),plane_overlap_para_v,label="  plane_overlap_para_v")
plt.plot(np.linspace(0,5,100),plane_overlap_dist_b,label="  plane_overlap_dist_b")
plt.plot(np.linspace(0,5,100),plane_overlap_dist_v,label="  plane_overlap_dist_v")
plt.plot(np.linspace(0,5,100),plane_overlap_res_b,label="   plane_overlap_res_b")
plt.plot(np.linspace(0,5,100),plane_overlap_res_v,label="   plane_overlap_res_v")  
plt.legend()
plt.show()
#'''
