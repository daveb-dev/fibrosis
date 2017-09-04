def add_gaussian_noise(h, tau = 0.002):  
  from dolfin import *  
  from functions.inout import readmesh
  import numpy as np
  import csv
  import math
  
  from functions import inout as io
  from functions import utils  

  """
  #================================
  # Load parameters from yaml file
  parameter_file	= './input/prms_add_gaussian_noise.yaml'
  prms 			= io.prms_load(parameter_file)
  
  T 			= prms['num']['T']
  dt 			= prms['num']['dt']
  mu 			= prms['num']['mu']
  noise_iterations 	= prms['num']['noise_iterations']

  results_dir 		= prms['io']['results']
  mesh_file 		= prms['io']['mesh']
  """
  dt = tau                    # time-sampling of input file
  T  = 0.544                      # total simulation time of input file
  mu = 0                       # mean for gaussian noise. The standar deviation is calculated using the sample
  mesh_file   = './meshes/COARTED_60_h' + str(h) +'.h5'
  results_dir = './results/velocity_measures/h' + str(h)
  input_file  = './results/velocity_measures/h' + str(h) + '/velocity.h5'

  #================================
  # Import mesh and declarse functions-space
  mesh, subdomains, boundaries = io.readmesh(mesh_file)
  V = VectorFunctionSpace(mesh, "CG", 1)
  ut, u0 = Function(V), Function(V)

  #================================
  # Load file to be perturbed and declare output files
  input_file = HDF5File(mesh.mpi_comm(), input_file, "r")
  u = Function(V)
  u_noise = Function(V)
  u_noise_vec = u_noise.vector()
  u_noise_file1 = HDF5File(mesh.mpi_comm(), results_dir + "/velocity_noise.h5", "w")
  u_noise_file2 = File(results_dir + "/velocity_noise_paraview.xdmf")

  rank = mesh.mpi_comm().Get_rank()
  t = 0.0
  print("Adding Gaussian Noise. \n \t -->  Time (seg) = [ "),
  import os  
  while t <= T:       
    print (str(t) + " "),
    input_file.read(u, "velocity_" + str(t))                  
    # The standard deviation is setted to the ten percent of the maxium velocity at the coarctation
    sigma = 0.1*250
    if sigma == 0:
      sigma = DOLFIN_EPS # add a minium value if sigma = 0 in order to get a well function of the gaussian generator

    u_vec = u.vector()

    noise = np.zeros(u_vec.local_size())    
    noise = np.random.normal(mu, sigma, float(u_vec.local_size()))
    
    u_noise_vec = u_vec.array() + noise
    u_noise.vector()[:] = u_noise_vec

    # Save solution to files
    u_noise_file1.write(u_noise, "velocity_" + str(t))
    u_noise_file2 << u_noise, t

    # Update measures function
    t += dt
  print "]\n"