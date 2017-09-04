import numpy as np
import matplotlib.pyplot as plt      
from solvers.pressure_drop import compute
from functions.add_gaussian_noise import add_gaussian_noise
import csv
import math
from functions import inout

if __name__ == "__main__":
    # PARAMETERS
    h          = 0.15
    tau        = 0.02
    noise      = 0
    T 	       = 0.5
    sigma      = .1*280

    # set-up methods to be calculated
    #TODO: pass this arguments to compute functions.
    # For now you have to change the parameters here and in the calculation module

    iterations = 100*noise + (1 - noise)

    aux_array = np.zeros((iterations, math.floor(T/tau) - 1, 9))

    # loop for noise-realizations    
    for k in range(iterations): # k is used to label pressure_drop results
      # Add gaussian noise to the velocity measures
      if noise:
	print "========================================\n\n\tNOISE REALIZATION # %g\n\n======================================== " % k
	#add_gaussian_noise(h, tau, sigma, k)
      time_array, aux_array[k,:,:] = compute(h, noise, tau, T, k)

    # Calculate mean pressure curve
    p_drop_array = np.mean(aux_array,0)
    if noise == 1:
      mean_file = open('./results/pressure_drop/mean_curve_h'+ str(h) +'_tau' + str(tau) + '.csv', 'w+')

    # Calculate standar deviation of the methods and save to file
    if noise == 1:
      std_dev		= np.std(aux_array, 0)
      std_dev_file = open('./results/pressure_drop/std/std_h' + str(h) +'_tau' + str(tau) + '.csv', 'w+')
      np.savetxt('./results/pressure_drop/std/std_h' + str(h) +'_tau' + str(tau) + '.csv', std_dev, delimiter=",")

    # Save standar deviation to file

    """
    # Import pressure drop reference (ground truth)     
    ref = np.zeros((math.floor(0.55/0.002), 2)); s = 0
    with open('./input/p_drop_ref.csv', 'rb') as csvfile:
      spamreader = csv.reader(csvfile, delimiter=';')
      for row in spamreader:
	ref[s,0], ref[s,1] = float(row[0])*1000, -1*float(row[1]); s+=1    

    # plot and save graphics
    def set_plot_options(noise):
      leg = plt.legend(loc='upper left')
      plt.xlabel('time [ms]')
      plt.ylabel('pressure drop [mmHg]')
      if noise == 1:
	plt.axis([1.5, 25.6, -5, 30])
      else:
	plt.axis([0, T*1000, -5, 30])
      plt.grid('on')  
      leg.get_frame().set_linewidth(0.0)

    def normalize(array, noise):
      if noise == 1:
	return 1.5 + array/(tau)
      else:
	return array

    labels = ["%.f" % x for x in time_array]    
    for i in range(time_array.size):
      if (i + 1)%2 == 0:	
	labels[i] = ' '

    # Change units
    h   = int(h*100)
    tau = int(tau*1000)

    if PIMRP:
      fig, ax1 = plt.subplots()
      plt.plot(normalize(ref[:,0], noise), ref[:,1], '-k' , linewidth=1.5, label = 'Ground truth')
      plt.plot(normalize(time_array, noise), p_drop_array[:,0], 'g', linewidth=0.8, label = 'PIMRP')
      if noise:
	plt.boxplot(aux_array[:,:,0], labels = labels,  showfliers=False)
      set_plot_options(noise)
      plt.savefig('../S_galarce_practicaprof/fig/new/PIMRP_h' + str(h) + '_tau' + str(tau) + '_noise' + str(noise) + '.eps')
      plt.savefig('../S_galarce_practicaprof/fig/new/PIMRP_h' + str(h) + '_tau' + str(tau) + '_noise' + str(noise) + '.png')      
      plt.show()

    if PPE:
      plt.plot(normalize(ref[:,0], noise), ref[:,1], '-k' , linewidth=1.5, label = 'Ground truth')
      plt.plot(normalize(time_array, noise), p_drop_array[:,3], '--r', linewidth = 1, label = 'PPE')
      if noise:
	plt.boxplot(aux_array[:,:,3], labels = labels,  showfliers=False)      
      
      if slow_mode:
	plt.plot(time_array, p_drop_array[:,4], '--g', linewidth = 2.5, label = 'PPE - P2')
	plt.plot(time_array, p_drop_array[:,5], '-b', linewidth = 1, label = 'PPE - P3')
	
      set_plot_options(noise)
      plt.savefig('../S_galarce_practicaprof/fig/new/PPE_h' + str(h) + '_tau' + str(tau) + '_noise' + str(noise) + '.eps')
      plt.savefig('../S_galarce_practicaprof/fig/new/PPE_h' + str(h) + '_tau' + str(tau) + '_noise' + str(noise) + '.png')
      plt.show()  

    if WERP:
      plt.plot(normalize(ref[:,0], noise)  , ref[:,1], '-k' , linewidth=1.5, label = 'Ground truth')
      plt.plot(normalize(time_array, noise), p_drop_array[:,6], '--g', linewidth=2, label = 'WERP')
      if noise:
	plt.boxplot(aux_array[:,:,6], labels = labels,  showfliers=False)      
      
      if noise == 1:
	plt.plot(normalize(time_array, noise), p_drop_array[:,7], '-r', linewidth=2, label = 'cWERP')
	plt.boxplot(aux_array[:,:,7], labels = labels,  showfliers=False)      
	
      set_plot_options(noise)
      plt.savefig('../S_galarce_practicaprof/fig/new/WERP_h' + str(h) + '_tau' + str(tau) + '_noise' + str(noise) + '.eps')
      plt.savefig('../S_galarce_practicaprof/fig/new/WERP_h' + str(h) + '_tau' + str(tau) + '_noise' + str(noise) + '.png')
      plt.show()

    if STE:
      plt.plot(normalize(ref[:,0], noise), ref[:,1], '-k' , linewidth=1.5, label = 'Ground truth')
      plt.plot(normalize(time_array, noise), p_drop_array[:,8] , 'r', linewidth = 1.5, label = 'STE')
      plt.plot(normalize(time_array, noise), p_drop_array[:,11], 'g', linewidth = 1, label = 'STEi - P1bP1')
      if noise:
	plt.boxplot(aux_array[:,:,8], labels = labels,  showfliers=False)
	plt.boxplot(aux_array[:,:,11], labels = labels,  showfliers=False)      
      
      if slow_mode:
	plt.plot(normalize(time_array, noise), p_drop_array[:,9] , '--g', linewidth = 2, label = 'STE - P2P1')
	plt.plot(normalize(time_array, noise), p_drop_array[:,10], '-b', linewidth = 1.5, label = 'STE - P3P2')            
	plt.plot(normalize(time_array, noise), p_drop_array[:,12], 'g', linewidth = 1, label = 'STEi - P2P1')
	plt.plot(normalize(time_array, noise), p_drop_array[:,13], 'b', linewidth = 1, label = 'STEi - P3P2')
	
      set_plot_options(noise)
      plt.savefig('../S_galarce_practicaprof/fig/new/STE_h' + str(h) + '_tau' + str(tau) + '_noise' + str(noise) + '.eps')
      plt.savefig('../S_galarce_practicaprof/fig/new/STE_h' + str(h) + '_tau' + str(tau) + '_noise' + str(noise) + '.png')
      plt.show()  

    if DAE:
      plt.plot(normalize(ref[:,0], noise), ref[:,1], '-k' , linewidth=1.5, label = 'Ground truth')      
      plt.plot(normalize(time_array, noise), p_drop_array[:,14], 'r', linewidth=1.5, label = 'DAE')
      plt.plot(normalize(time_array, noise), p_drop_array[:,17], 'g', linewidth=1, label = 'DAEi')
      if noise:
	plt.boxplot(aux_array[:,:,14], labels = labels,  showfliers=False)      
	plt.boxplot(aux_array[:,:,17], labels = labels,  showfliers=False)            
     
      if slow_mode:
	plt.plot(normalize(time_array, noise), p_drop_array[:,15], '--g', linewidth=1.8, label = 'DAE  - RT2P2')            
	plt.plot(normalize(time_array, noise), p_drop_array[:,18], 'g', linewidth=1, label = 'DAEi - RT2P2')
	
      set_plot_options(noise)
      plt.savefig('../S_galarce_practicaprof/fig/new/DAE_h' + str(h) + '_tau' + str(tau) + '_noise' + str(noise) + '.eps')
      plt.savefig('../S_galarce_practicaprof/fig/new/DAE_h' + str(h) + '_tau' + str(tau) + '_noise' + str(noise) + '.png')
      plt.show()
    """