# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def plot_data(data,x_value,number=300):
    values   = data[:number]
    values_x = values[np.arange(0,number,3)]
    values_y = values[np.arange(1,number,3)]
    values_z = values[np.arange(2,number,3)]

    errors   = data[number:]
    errors_x = errors[np.arange(0,number,3)]
    errors_y = errors[np.arange(1,number,3)]
    errors_z = errors[np.arange(2,number,3)]

    f,ax=plt.subplots(1)
    ax.errorbar(x_value,values_x,yerr=errors_x,fmt="o",label="x-Richtung")
    ax.errorbar(x_value,values_y,yerr=errors_y,fmt="o",label="y-Richtung")
    ax.errorbar(x_value,values_z,yerr=errors_z,fmt="o",label="z-Richtung")
    return ax




#flow error
#-----------------------------------------------
flow= plot_data(np.load("effect_of_flow_errors.npy"),0.001*np.linspace(0,100,100))
flow.set_xlabel("Fehler in Fluss und Position [m]",fontsize=24)
flow.set_ylabel("Geschwindigkeit [m/s]",fontsize=24)
plt.tick_params(labelsize=20)
plt.legend(prop={'size': 20})
plt.xkcd()
plt.show()


#distance error
#------------------------------------------------
dist= plot_data(np.load("effect_of_distance_error.npy"),0.001*np.linspace(0,100,100))
dist.set_xlabel("Fehler in der Distanz zum Boden [m]",fontsize=24)
dist.set_ylabel("Geschwindigkeit [m/s]",fontsize=24)
plt.tick_params(labelsize=20)
plt.legend(prop={'size': 20})
plt.show()

#ang_vel_err
#------------------------------------------------
ang= plot_data(np.load("effect_o_ang_vel_error.npy"),0.001*np.linspace(0,100,100))
ang.set_xlabel("Fehler in der Drehgeschwindigkeit [rad/s]",fontsize=24)
ang.set_ylabel("Geschwindigkeit [m/s]",fontsize=24)
plt.tick_params(labelsize=20)
plt.legend(prop={'size': 20})
plt.show()

#normal err
#------------------------------------------------
norm= plot_data(np.load("effect_of_normal_error.npy"),0.001*np.linspace(0,100,100))
norm.set_xlabel("Fehler im Normalvektor",fontsize=24)
norm.set_ylabel("Geschwindigkeit [m/s]",fontsize=24)
plt.tick_params(labelsize=20)
plt.legend(prop={'size': 20})
plt.show()

#translation err
#------------------------------------------------
tran= plot_data(np.load("effect_of_translation_error.npy"),0.001*np.linspace(0,100,100))
tran.set_xlabel("Fehler in der Translation [m]",fontsize=24)
tran.set_ylabel("Geschwindigkeit [m/s]",fontsize=24)
plt.tick_params(labelsize=20)
plt.legend(prop={'size': 20})
plt.show()

#orientation
#----------------------------------------------------------------
orie= plot_data(np.load("effect_of_orientation.npy"),180*np.linspace(0,100,100)/100)
orie.set_xlabel("Winkel zur Senkrechten [Grad]",fontsize=24)
orie.set_ylabel("Geschwindigkeit [m/s]",fontsize=24)
plt.tick_params(labelsize=20)
plt.legend(prop={'size': 20})
plt.show()

#hoehe
#-----------------------------------------------------------------
heig= plot_data(np.load("effect_of_height.npy"),np.linspace(0.2,7.65,100))
heig.set_xlabel("Distanz zum Boden [m]",fontsize=24)
heig.set_ylabel("Geschwindigkeit [m/s]",fontsize=24)
plt.tick_params(labelsize=20)
plt.legend(prop={'size': 20})
plt.show()


#position
#-----------------------------------------------------------------
posi= plot_data(np.load("effect_of_point_position.npy"),np.linspace(0,100,100)/100)
posi.set_xlabel("Mittlere Featureposition [m]",fontsize=24)
posi.set_ylabel("Geschwindigkeit [m/s]",fontsize=24)
plt.tick_params(labelsize=20)
plt.legend(prop={'size': 20})
plt.show()

#numerical
#-----------------------------------------------------------------
nume= plot_data(np.load("number_of_point.npy"),2+2*np.linspace(0,99,99),number=297)
num_err=np.load("sim_err_vs_num2.npy")
nume.errorbar(2+2*np.linspace(0,99,99),np.ones(99),num_err,label="Numerischer Fehler",zorder=1)
nume.set_xlabel("# Features",fontsize=24)
nume.set_ylabel("Geschwindigkeit [m/s]",fontsize=24)
plt.tick_params(labelsize=20)
plt.legend(prop={'size': 20})
plt.show()

#points
#-----------------------------------------------------------------
heig= plot_data(np.load("number_of_point.npy"),2+2*np.linspace(0,99,99),number=297)
heig.set_xlabel("# Features ",fontsize=24)
heig.set_ylabel("Geschwindigkeit [m/s]",fontsize=24)
plt.tick_params(labelsize=20)
plt.legend(prop={'size': 20})
plt.show()


