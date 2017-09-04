''' Pressure drop estimator in coarted blood vessels
    author: Felipe Galarce
    email: felipe.galarce.m@gmail.com  '''

from dolfin import *
from functions.inout import readmesh

import matplotlib.pyplot as plt
import numpy as np
import csv

#============================
# Load parameters
T 	= 0.4
tau	= 0.005
h	= 0.06
coart	= 60

mesh_file 	= './meshes/COARTED_60_h' + str(h) + '.h5'
press_file	= './results/Navier_Stokes/p1bp1_' + str(h) + '_COART'+ str(coart) +'/pressure.h5'
press_drop_file = open("./results/p_drop_ref.csv", 'w+')

#============================
# Import mesh and create space functions to import solution
mesh, subdomains, boundaries = readmesh(mesh_file)
Q  	= FunctionSpace(mesh, "CG",  1)
p	= Function(Q)

#============================
# Load pressure data
p_file 	= HDF5File(mesh.mpi_comm(), press_file, "r")
ds	= Measure('ds', domain=mesh  , subdomain_data=boundaries)
t, k	= 0,0

p_drop = []
time = []

print "Calculating pressure drop..."
while t <= T + tau:
  print t,
  p_file.read(p, "/pressure_" + str(t))
  time.append(t*1000)
  p_drop.append(-(assemble(p*ds(1)) - assemble(p*ds(0)))/DOLFIN_PI*0.000750061505)
  # save pressure drop to file
  press_drop_file.write(str(t) + "; " + str(p_drop) + "; \n")
  t += tau
  k += 1

# plot and save vectorial file
plt.plot(time, p_drop, '-b' , linewidth=2.5)
plt.xlabel('time [ms]')
plt.ylabel('pressure drop [mmHg]')
plt.axis([0,T*1000, -35, 5])
plt.grid('on')
# save plot
plt.savefig('Reference_pdrop.eps')
plt.show()
p_file.close()