from dolfin import *
import numpy as np


def compute(bval, param):
    ''' Bloch equation solver   LEGACY
        author: David Nolte
        email: dnolte@dim.uchile.cl
    '''

    from functions.inout import readmesh

    # Extract parameters
    mesh_file = param['io']['mesh']
    results_dir = param['io']['results']

    dt = param['num']['dt']
    sol, prec = param['num']['krylov']
    deg = param['num']['deg']

    f = param['phys']['pseq']['f']
    Gdir = param['phys']['pseq']['dir']
    dtp = param['phys']['pseq']['dt']
    k = param['phys']['D']

    h5fname = "b"+str(int(bval)).zfill(4)  # +"_ts.h5"

    # Parallelize for shared memory multicore machines
    # Multithread assembly of matrices
    # parameters['num_threads'] = 4

    # Mesh partitioner for parallel distributed systems
    parameters['mesh_partitioner'] = 'ParMETIS'  # default: SCOTCH
    # NOTE: parmetis was optimized on the NLHPC cluster

    # Physical constants    NOTE: better all to SI units?
    pi = DOLFIN_PI
    gm = 2*pi*42.57e3      # (Hz/mT) gyromagnetic ratio H1

    # calculate gradient strength for spec. sequence and b-value
    # see for example Bernstein (2004), p.279
    G = sqrt(bval/(gm**2*dtp[0]**2*(dtp[1]-dtp[0]/3.)))  # mT/mm

    # end time of simulation
    T = sum(dtp)

    if param['num']['dry'] == True:
        # perform DRY RUN (e.g. if FFC fails on cluster)
        # disable time stepping and vtk output of initial condition
        T = -1
        param['io']['vtk'] = 0
        print("PERFORMING DRY RUN")

    # Read mesh, boundaries and subdomains
    mesh, subdomains, boundaries = readmesh(mesh_file)

    # Define vectorial function space (2D)
    V = FunctionSpace(mesh, "CG", deg)
    W = V*V

    dx = Measure('dx', domain=mesh, subdomain_data=subdomains)

    # Initial condition
    I = Constant((1.0, 0.0))
    ui = project(I, W)       # TODO interpolate??
    u1 = ui.copy()           # u^k-1

    # BC: start with Neumann BC 0 on all boundaries (full reflective)

    # Define functions for variational form
    u = TrialFunction(W)
    v = TestFunction(W)

    # Gradient matrix operator, time independent
    Gr = Expression((('0', '-G*(x[0]*r0+x[1]*r1+x[2]*r2)'),
                     ('G*(x[0]*r0+x[1]*r1+x[2]*r2)', '0')), G=G, r0=Gdir[0],
                    r1=Gdir[1], r2=Gdir[2])

    # Convert diffusion coefficients to UFL constants
    # k1 = Constant(D1)
    # k2 = Constant(D2)

    # Implicite backward step
    # NOTE: The forms a(u,v) and L(v) are time dependent. L via u1 (previous
    #       time step) and a(u,v) via f(t). Avoid repeated assembly!
    #       f(t) changes 2x but is constant. a) Split time loop into
    #       corresponding intervals. b) More general (arbitrary sequences,
    #       e.g., oscillating): pre-assemble A and b. (b) implemented!)
    # (http://fenicsproject.org/documentation/tutorial/timedep.html)

    # ASSEMBLE MATRICES
    a_M = inner(u, v)*dx
    a_N = - gm*inner(dot(Gr, u), v)*dx
    sdix = np.unique(subdomains.array())
    print(sdix)
    a_K = inner(Constant(k[0])*grad(u), grad(v))*dx(int(sdix[0]))
    for i in range(1, len(sdix)):
        a_K += inner(Constant(k[i])*grad(u), grad(v))*dx(i)
    # a_K = inner(k1*grad(u), grad(v))*dx(0) + inner(k2*grad(u), grad(v))*dx(1)

    M = assemble(a_M)
    N = assemble(a_N)
    K = assemble(a_K)

    A0 = M + dt*K
    N *= dt

    # Time loop
    t = 0
    w = Function(W, name="M")
    w.assign(ui)

    # TODO: output function
    if param['io']['vtk'] == 1:
        ufile = File(results_dir+"/u_b"+str(int(bval)).zfill(4)+".pvd")
        ufile << w

    if param['io']['timeseries'] == 1:
        # compute signal = integral of M over the domain
        # initialize appropiate function space and assemble
        # R = FunctionSpace(mesh, "R", 0)
        # RR = R*R
        # c = TestFunction(RR)
        # m = assemble(dot(w, c)*dx).array()
        # faster:
        m = [assemble(w[0]*dx), assemble(w[1]*dx)]
        # mabs = assemble(dot(abs(w), c)*dx).array()

        # TODO: do I need abs(M)?
        ml = []
        # mlabs = []
        tl = []
        # add initial condition
        tl.append(0)
        ml.append(m)

    # Init Krylov solver
    solver = KrylovSolver(sol, prec)

    rank = mesh.mpi_comm().Get_rank()
    T = 0.005
    while t < T:
        t += dt
        A = A0 + f(t, dtp)*N
        b = M*u1.vector()
        solver.solve(A, w.vector(), b)
        u1.assign(w)

        if rank == 0:
            print(t)

        if param['io']['timeseries'] == 1:
            # compute signal = integral of M over the domain
            # m = assemble(dot(w, c)*dx).array()
            # faster:
            m = [assemble(w[0]*dx), assemble(w[1]*dx)]
            # mabs = assemble(dot(abs(w), c)*dx).array()
            ml.append(m)
            # mlabs.append(mabs)
            tl.append(t)

        if param['io']['vtk'] == 1:
            ufile << w

    if param['io']['timeseries'] == 1 and not param['num']['dry']:
        import h5py
        with h5py.File(results_dir+"/"+h5fname+"_ts_func.h5", "w",
                       comm=mesh.mpi_comm()) as h5:
            h5.create_dataset('time', data=tl)
            h5.create_dataset('u', data=ml)
            # h5.create_dataset('m_abs', data=mlabs)
            h5.attrs['b'] = bval
            h5.close()

    S_TE = None
    if param['io']['signal_TE'] == 1:
        if param['io']['timeseries'] == 1:
            S_TE = ml[-1]
        else:
            # R = FunctionSpace(mesh, "R", 0)
            # RR = R*R
            # c = TestFunction(RR)
            # S_TE = assemble(dot(w, c)*dx).array()
            S_TE = [assemble(w[0]*dx), assemble(w[1]*dx)]

    return S_TE
