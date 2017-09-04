''' solver module '''
from dolfin import *
import numpy as np

def compute(mu = 0.35):
    ''' Stokes stationary flow equation solver
        author: Felipe Galarce
        email: felipe.galarce.m@gmail.com
    '''

    sol   		= 'mumps'                       # Krylov solver settings
    prec  		= 'none'                        # Preconditioner (use 'none' to not use preconditioner)
    h     		= 0.1
    ST_fspace_order	= 'P1bP1'
    rho   		= 1
    coart 		= 60

    nu = mu/rho
    #parameters["form_compiler"]["quadrature_degree"] = 6

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

    results_dir = '../results/test_functions/brinkman_h' + str(h) + '_mu' + str(mu) + '_' + ST_fspace_order

    from functions.inout import readmesh
    # Read mesh, boundaries and subdomains
    mesh, subdomains, boundaries_MALO = readmesh('../meshes/COARTED_' + str(coart) + '_h' + str(h) + '.h5')


    # ******************************************
    # Initialize sub-domain instances
    inlet 	 = Inlet()
    outlet 	 = Outlet()
    noslip_bound = No_slip()

    # Mark boundaries of the domain
    boundaries = FacetFunction("size_t", mesh)
    boundaries.set_all(3)
    noslip_bound.mark(boundaries, 2)
    inlet.mark(boundaries, 1)
    outlet.mark(boundaries, 0)

    # Define function spaces
    if ST_fspace_order == 'P1bP1':
      P1 = VectorFunctionSpace(mesh, "Lagrange", 1)
      B  = VectorFunctionSpace(mesh, "Bubble"  , 4)
      V  = P1 + B
      Q  = FunctionSpace(mesh, "CG",  1)

    if ST_fspace_order == 'P2P1':
      V = VectorFunctionSpace(mesh, "Lagrange" , 2)
      Q = FunctionSpace(mesh, "CG",  1)

    if ST_fspace_order == 'P3P2':
      V = VectorFunctionSpace(mesh, "Lagrange" , 3)
      Q = FunctionSpace(mesh, "CG",  2)

    Mini = V*Q

    # Mark UFL variable with boundaries
    ds   = Measure('ds', domain=mesh  , subdomain_data=boundaries)

    # Define variational problem
    (u, p) = TrialFunctions(Mini)
    (v, q) = TestFunctions(Mini)
    f = Constant((0, 0, 0))
    gamma = pow(10, 10)

    nu = Expression('nu', nu = nu)
    n	= FacetNormal(mesh)
    a	= (inner(u, v) + nu*inner(grad(u), grad(v)) + p*div(v) + q*div(u))*dx + float(gamma)*inner(u, n)*inner(v, n)*ds(2) - nu*inner(v, grad(u)*n)*ds(0) - nu*inner(v, grad(u)*n)*ds(1) - nu*inner(v, grad(u)*n)*ds(2)    
    L	= inner(f, v)*dx

    bc_in  = DirichletBC(Mini.sub(0), project(Constant((0, 1, 0)), V), boundaries, 1)
    bc_out = DirichletBC(Mini.sub(0), project(Constant((0, 1, 0)), V), boundaries, 0)
    bcs = [bc_in, bc_out]

    if has_petsc():
      PETScOptions.set("mat_mumps_icntl_14", 40.0)

    # Compute solution
    w = Function(Mini)
    rank = mesh.mpi_comm().Get_rank()
    A = PETScMatrix()
    b = PETScVector()
    A, b = assemble_system(a, L, bcs)
    solve(a == L, w, bcs)

    # Split the mixed solution using a shallow copy
    (u, p) = w.split()

    plot(u);  plot(p); interactive()

    # Save solution to file (to be read as weighting function in pressure drop estimation)
    u_file = HDF5File(mesh.mpi_comm(), results_dir + "/velocity.h5", "w")
    p_file = HDF5File(mesh.mpi_comm(), results_dir + "/pressure.h5", "w")
    u_file.write(u, "velocity")
    p_file.write(p, "pressure")
    u_file.close(); u_file.close();

    # Save solution to file (to be read in paraview)
    ufile_pvd = File(results_dir + "/velocity_paraview.xdmf");  ufile_pvd << u
    pfile_pvd = File(results_dir + "/pressure_paraview.xdmf");  pfile_pvd << p

'''
mu = []
for i in range(5,0,-1):
  mu.append(0.035*pow(10, -i))

for i in range(5):
  mu.append(0.035*pow(10, i))

for i in range(len(mu)):
    print "solving for mu = %g poise" % mu[i]
    compute(mu[i])
'''
compute()
