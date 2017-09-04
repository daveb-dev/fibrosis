''' solver module '''
from dolfin import *

def compute():
    ''' Navier-Stokes flow equation solver
        author: Felipe Galarce
        email: felipe.galarce.m@gmail.com
    '''
    from functions.inout import readmesh
    """
    # Extract parameters    
    results_dir = prms['io']['results']

    dt		= prms['num']['dt']
    T		= prms['num']['T']
    h		= prms['num']['h']
    coart	= prms['num']['coart']

    mu		= prms['phys']['mu']
    rho		= prms['phys']['rho']
    """
    # Configuring form compiler    
    parameters["form_compiler"]["optimize"] = True
    parameters["form_compiler"]["representation"] = 'quadrature'
    parameters["form_compiler"]["quadrature_degree"] = 6

    dt    = 0.005
    T     = 0.5
    h     = 0.06
    coart = 60
    mu    = 0.035
    rho   = 1
    results_dir = './results/Navier_Stokes/p1bp1_' + str(h) + '_COART' + str(coart)

    # Read mesh, boundaries and subdomains
    mesh, subdomains, boundaries_BAD = readmesh("./meshes/COARTED_" + str(coart) + "_h" + str(h) + ".h5")    

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


    # ******************************************
    # Initialize sub-domain instances
    inlet 		= Inlet()
    outlet 		= Outlet()
    noslip_bound    = No_slip()


    # Mark boundaries of the domain
    boundaries = FacetFunction("size_t", mesh)
    boundaries.set_all(3)
    noslip_bound.mark(boundaries, 2)
    inlet.mark(boundaries, 1)
    outlet.mark(boundaries, 0)

    # Define function spaces
    P1 	= VectorFunctionSpace(mesh, "CG", 1)
    B  	= VectorFunctionSpace(mesh, "Bubble"  , 4)
    V	= P1 + B    
    #V 	= VectorFunctionSpace(mesh, "CG", 2)
    Q  	= FunctionSpace(mesh, "CG",  1)
    Mini = V*Q
    
    # No-slip boundary condition for velocity
    noslip = project(Expression(("0.0","0.0", "0.0"), degree = 3), V)
    bc0 = DirichletBC(Mini.sub(0), noslip, boundaries, 2)

    # INFLOW bc
    inflow = Expression(("0.0","-60*(x[0]*x[0] + x[2]*x[2] - 1)*sin(5*DOLFIN_PI/2*t)", "0.0"), t = 0)

    # Characterize boundaries
    ds   = Measure('ds', domain=mesh  , subdomain_data=boundaries)

    w0 = Function(Mini)
    p0 = Function(Q)
    u0 = project(inflow, V)

    (u, p) = TrialFunctions(Mini)
    (v, q) = TestFunctions(Mini)
    #print " -- > Ensamblando matriz de masa"
    #M     = assemble(inner(u, v)*dx)
    #print " -- > Ensamblando matriz de rigidez"
    #K     = assemble(inner(grad(u), grad(v))*dx)
    #print " -- > Ensamblando matriz B"
    #B     = assemble(q*div(u)*dx)
    #print " -- > Ensamblando matriz Bt"
    #Bt    = assemble(div(v)*p*dx)

    # Variatonal forms (slow version without pre-assembly)
    a = rho/dt*inner(u,  v)*dx + mu*inner(grad(u), grad(v))*dx + q*div(u)*dx - div(v)*p*dx + rho*inner(grad(u)*u0, v)*dx + 0.5*rho*(div(u0)*inner(u, v))*dx
    L = rho/dt*inner(u0, v)*dx


    # files to save solution
    u_file = HDF5File(mesh.mpi_comm(), results_dir + "/velocity.h5", "w")
    p_file = HDF5File(mesh.mpi_comm(), results_dir + "/pressure.h5", "w")
    ufile = File(results_dir + "/velocity_paraview.xdmf")
    pfile = File(results_dir + "/pressure_paraview.xdmf")

    # Save initial condition to file    
    u_file.write(u0, "velocity_0")
    p_file.write(p0, "pressure_0")
    ufile << u0, 0
    pfile << p0, 0

    # Configure solver            
    solver = LinearSolver('mumps')        
    # Set PETSc MUMPS paramter (this is required to prevent a memory error
    # in some cases when using MUMPS LU solver).
    if has_petsc():
      PETScOptions.set("mat_mumps_icntl_14", 40.0)

    w = Function(Mini)
    rank = mesh.mpi_comm().Get_rank()
    t = dt

    while t <= T + dt:
      if rank == 0:
	print 'time: %g \t [sec]' % t
      if rank == 0:
	print "--> Interpolando condiciones de borde en \Gamma_in"
      inflow = project(Expression(("0.0","-60*(x[0]*x[0] + x[2]*x[2] - 1)*sin(5*DOLFIN_PI/2*t)", "0.0"), t = t,degree = 3), V)
      bc1 = DirichletBC(Mini.sub(0), inflow, boundaries, 1)
      bcs = [bc0, bc1]

      if rank == 0:
	print "--> Ensamblando sistema"
      A, b = assemble_system(a, L, bcs)      
      if rank == 0:
	print "--> Resolviendo sistema lineal mediante MUMPS"
      solver.solve(A, w.vector(), b)
      ut, pt = w.split(deepcopy = True)

      pfile << pt, t
      ufile << ut, t
      # Save solution to file (to calculate pressure drop)
      u_file.write(ut, "velocity_" + str(t))
      p_file.write(pt, "pressure_" + str(t))   
      """
      plot(pt, title = 'pressure', rescale = True)
      plot(ut, title = 'velocity', rescale = True)
      interactive()
      """
      u0.assign(ut)
      p0.assign(pt)
      t += dt
    u_file.close(); p_file.close();