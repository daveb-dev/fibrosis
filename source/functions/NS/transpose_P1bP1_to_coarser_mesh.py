# Interpolate solution between different finite-element spaces
# Nore that this script ONLY WORKS IN SERIAL, due to the interpolation parallel issue in FEniCS

from dolfin import *
from functions.inout import readmesh
import numpy as np

# ================================
# Parameters   
h_in 			= 1
h_out			= 1
mesh_in_file 		= './meshes/CVV_60_h' + str(h_in)  + '.h5'
mesh_out_file		= "./meshes/CVV_60_h" + str(h_out) + ".h5"
reference_solution 	= './results/trial_functions/h' + str(h_in) + '/velocity.h5'
results_dir		= './results'
T			= 0.6
tau			= 0.002

# ================================
# Import meshes
mesh, subdomains, boundaries 			= readmesh(mesh_in_file)
mesh_out, subdomains_out, boundaries_out 	= readmesh(mesh_out_file)

# ================================
# Define function spaces and functions
P1 = VectorFunctionSpace(mesh, "Lagrange", 1)
B  = VectorFunctionSpace(mesh, "Bubble", 4)
V  = P1 + B
V_out =  VectorFunctionSpace(mesh_out, "Lagrange", 1)
u  = Function(V)

# ================================
# Load Stationary Stokes Solution 
u_file = HDF5File(mesh.mpi_comm(), results_dir + "/test_functions/stokes_h" + str(h_in) + "/velocity.h5", "r")
u_file.read(u, "/velocity")

# ================================
# Save interpolated stokes solution to File
noslip = interpolate(Expression(("0.0","0.0", "0.0"), degree=3), V_out)
bc = DirichletBC(V_out, noslip, boundaries_out, 2)
u_P1 = interpolate(u, P1)
u_P1.set_allow_extrapolation(True)
u_P1 = interpolate(u_P1, V_out)
bc.apply(u_P1.vector())


u_out1 	= HDF5File(mesh.mpi_comm(), results_dir + "/test_functionsASD/stokes_h" + str(h_out) + "/velocity.h5", "w")
u_out2 	= File(results_dir + "/test_functionsASD/stokes_h" + str(h_out) + "/velocity_paraview.xdmf")

u_out1.write(u_P1, "velocity")
u_out2 << u_P1
u_out1.close(); 


# ================================
# Load Navier-Stokes sinthetic measures
u_file = HDF5File(mesh.mpi_comm(), results_dir + "/measures/h" + str(h_in) + "/velocity.h5", "r")
u_out1 		= HDF5File(mesh.mpi_comm(), results_dir + "/measuresASD/h" + str(h_out) + "/velocity.h5", "w")
u_out2 		= File(results_dir + "/measuresASD/h" + str(h_out) + "/velocity_paraview.xdmf") 

u_file.read(u, "/velocity_0")

t = 0
rank = mesh.mpi_comm().Get_rank()

while t <= T:
  if rank == 0:
    print 'time: %g' % t

  # Interpolate in coarser mesh
  u_P1 = interpolate(u, P1)
  u_P1.set_allow_extrapolation(True) 
  u_P1 = interpolate(u_P1, V_out)
  bc.apply(u_P1.vector())

  # Save solution to files
  u_out1.write(u_P1, "velocity_" + str(t))
  u_out2 << u_P1

  # update measures
  t += tau
  u_file.read(u, "velocity_" + str(t))
