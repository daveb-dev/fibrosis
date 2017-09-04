# Interpolate solution between different finite-element spaces
# Nore that this script ONLY WORKS IN SERIAL, due to the interpolation parallel issue in FEniCS

from dolfin import *
from functions.inout import readmesh
import numpy as np

# ================================
# Parameters   
h_in 			= 0.06
h_out			= 0.15
mesh_in_file 		= './meshes/COARTED_60_h' + str(h_in)  + '.h5'
mesh_out_file		= './meshes/COARTED_60_h' + str(h_out) + '.h5'
reference_solution 	= './results/Navier_Stokes/p1bp1_' + str(h_in) + '/velocity.h5'
results_dir		= './results/velocity_measures/h' + str(h_out)
T			= 0.546
tau			= 0.002

# ================================
# Import meshes
mesh, subdomains, boundaries 		 = readmesh(mesh_in_file)
mesh_out, subdomains_out, boundaries_out = readmesh(mesh_out_file)

# ================================
# Define function spaces and functions
P1 = VectorFunctionSpace(mesh, "Lagrange", 1)
B  = VectorFunctionSpace(mesh, "Bubble", 4)
V  = P1 + B
V_out =  VectorFunctionSpace(mesh_out, "Lagrange", 1)
u  = Function(V)

# ================================
# Load Navier-Stokes sinthetic measures
u_file = HDF5File(mesh.mpi_comm(), reference_solution, "r")
u_out1 		= HDF5File(mesh.mpi_comm(), results_dir + "/velocity.h5", "w")
u_out2 		= File(results_dir + "/velocity_paraview.xdmf") 

u_file.read(u, "/velocity_0")

t = 0.0
rank = mesh.mpi_comm().Get_rank()

while t < T:
  print 'time: %g \t [seg]' % t
  if h_in != h_out:
    # Interpolate in coarser mesh  
    u.set_allow_extrapolation(True) 
    u_P1 = interpolate(u, P1)    
    u_P1.set_allow_extrapolation(True) 
    u_P1 = interpolate(u_P1, V_out)  
  else:  
    u.set_allow_extrapolation(True) 
    u_P1 = interpolate(u, P1)    
    
  # Save solution to files
  u_out1.write(u_P1, "velocity_" + str(t))
  u_out2 << u_P1

  # update measures
  t += tau
  u_file.read(u, "velocity_" + str(t))
