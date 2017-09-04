''' solver module '''
from dolfin import *

def compute():
    ''' Stokes transient flow equation solver
        author: Felipe Galarce
        email: felipe.galarce.m@gmail.com
    '''

    from functions.inout import readmesh
    
    dt  = 0.01
    T   = 5

    mu  = 0.035                                   # dynamic viscocity
    rho = 1                                      # density
    h   = 0.1
    mesh_file   = './meshes/COARTED_60_h' + str(h) + '.h5'     # mesh file
    results_dir = './results/stokes_tran'                            # results directory
    
    # Read mesh, boundaries and subdomains
    mesh, subdomains, boundaries = readmesh(mesh_file)

    # Define function spaces
    P1 = VectorFunctionSpace(mesh, "CG", 1)
    B  = VectorFunctionSpace(mesh, "Bubble"  , 4)
    Q  = FunctionSpace(mesh, "CG",  1)
    V = P1 + B    
    Mini = V*Q

    # No-slip boundary condition for velocity
    noslip = project(Constant((0, 0, 0)), V)
    bcs = DirichletBC(Mini.sub(0), noslip, boundaries, 2)

    # Characterize boundaries
    ds   = Measure('ds', domain=mesh  , subdomain_data=boundaries)
    #n = Expression(('0.0', '-1.0', '0.0'))
    n = FacetNormal(mesh)
  
    # Define variational problem
    p_in = Expression('sin(2*DOLFIN_PI*t)', t = 0)
    (u, p) = TrialFunctions(Mini)
    (v, q) = TestFunctions(Mini)

    # initial condition
    class VolumeInitialCondition(Expression):
      def eval(self, value, x):
	value[0] = 0
	value[1] = 0
	value[2] = 0
	value[3] = 0
      def value_shape(self):
	return (4,)
    w0 =  project(VolumeInitialCondition(), Mini)
    u0, p0 = w0.split(deepcopy = True)

    a = rho*inner(u, v)*dx + dt*(mu*inner(grad(u), grad(v)) - div(v)*p + q*div(u))*dx
    L = rho*inner(u0, v)*dx + dt*inner(v, n)*p_in*ds(1)

    A = assemble(a)

    # Compute solution
    w = Function(Mini)
    rank = mesh.mpi_comm().Get_rank()

    ufile = File(results_dir + "/velocity.xdmf")
    pfile = File(results_dir + "/pressure.xdmf")

    rank = mesh.mpi_comm().Get_rank()    
    t = dt

    while t < T:
      if rank == 0:
	print 'time: %g [sec]' % t
      p_in.t = t

      ut, pt = w.split(deepcopy = True)
      u0.assign(ut)
      p0.assign(pt)

      # TODO: speed up this splitting lhs
      b = assemble(L)
      bcs.apply(A,b)
      solve(A, w.vector(), b, 'mumps')

      plot(u0)
      interactive()

      # Save solution in VTK format
      pfile << p0, t
      ufile << u0, t
      t += dt

    # SAVE final solution to be used as initial condition for another problem
    File('saved_u.xml') << u0