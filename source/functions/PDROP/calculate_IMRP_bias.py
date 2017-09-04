from functions.inout import readmesh
from dolfin import *

import sys
sys.setrecursionlimit(9999999)
# Calculate IMRP bias

# Class with subdomains to mark borders of the domain
class Inlet(SubDomain):
    def inside(self, x, on_boundary):
      return x[1] < DOLFIN_EPS
class Outlet(SubDomain):
    def inside(self, x, on_boundary):
	return x[1] > 5 - 5*DOLFIN_EPS
class No_slip(SubDomain):
    def inside(self, x, on_boundary):
	return on_boundary and x[1] <= 5 - DOLFIN_EPS and x[1] >= 0.0

h		= 0.25
rho		= 1
sigma		= 28
test_functions  = './results/test_functions'

mesh, subdomains, boundaries_MALO = readmesh('./meshes/COARTED_60_h' + str(h) + '.h5')

# ******************************************
# Initialize sub-domain instances
inlet 	= Inlet()
outlet 	= Outlet()
noslip_bound  = No_slip()

# Mark boundaries of the domain
boundaries = FacetFunction("size_t", mesh)
boundaries.set_all(3)
noslip_bound.mark(boundaries, 2)
inlet.mark(boundaries, 1)
outlet.mark(boundaries, 0)

ds, n = Measure('ds', domain=mesh  , subdomain_data=boundaries), FacetNormal(mesh)

P1 = VectorFunctionSpace(mesh, 'Lagrange',     1)
P2 = VectorFunctionSpace(mesh, 'Lagrange', 2)
B  = VectorFunctionSpace(mesh, 'Bubble', 4)
V  = P1 + B


P1_dim1 = FunctionSpace(mesh, 'Lagrange', 1)
P2_dim1 = FunctionSpace(mesh, 'Lagrange', 2)
B_dim1  = FunctionSpace(mesh, 'Bubble', 4)
V_dim1  = P1_dim1 + B_dim1

# Calculate alpha
alpha = Function(P2_dim1)
ff    = Function(P1_dim1)
ndofs = ff.vector().array().size

for i in range(20):
  print 'looping over dof # %g of %g' % (i, ndofs)
  ff_vec         = ff.vector()
  ff_vec_arr     = ff_vec.array()
  ff_vec_arr[i]  = 1                # ff is the shape function of the i-th node
  ff.vector()[:] = ff_vec_arr

  tmp =  interpolate(ff, P2_dim1)
  tmp_vec = tmp.vector()
  tmp_vec_arr = tmp_vec.array()
  tmp_vec_arr = tmp_vec_arr*tmp_vec_arr
  print tmp_vec_arr

  alpha_vec = alpha.vector()
  alpha_arr = alpha_vec.array()
  alpha_arr = alpha_arr + tmp_vec_arr
  alpha.vector()[:] = alpha_arr

  ff_vec         = ff.vector()
  ff_vec_arr     = ff_vec.array()
  ff_vec_arr[i]  = 0
  ff.vector()[:] = ff_vec_arr

# Load stokes functions
v  = Function(V)
u_st_file = HDF5File(mesh.mpi_comm(), test_functions + "/stokes_h" + str(h) + "_P1bP1/velocity.h5", "r")
u_st_file.read(v, "/velocity")



v.set_allow_extrapolation(True)
v = interpolate(v, P2)

# Save function to file 
File('alpha_' + str(h) + '.xml') << alpha


lambda_v = assemble(inner(v,  n)*ds(1))
n    = FacetNormal(mesh)
BIAS = rho*sigma*sigma/(2*lambda_v)*(-assemble(alpha*div(v)*dx) + assemble(alpha*inner(v, n)*ds(0) + alpha*inner(v, n)*ds(1) + alpha*inner(v, n)*ds(2)))

print "IMRP BIAS = %g [CGS]" % BIAS
print float(BIAS)