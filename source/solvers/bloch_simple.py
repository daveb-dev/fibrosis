from dolfin import *
import numpy as np


def initialize(param, bval):
    ''' initialize and package variables in dictionaries for cleanness
        input:
            param:  dictionary with case setup
            bval:   b-value of the experiment
        return:
            meshdata:   dictionary of mesh, subdomains, boundaries
            ppdata:     dictionary with post processing settings
            solver:     initialized LS solver
    '''
    from functions import inout, utils
    mesh, sd, bnd = inout.read_mesh(param['io']['mesh'])

    # meshdata dict
    meshdata = {'mesh': mesh, 'sd': sd, 'bnd': bnd}

    # invoke solver
    sol, prec = param['num']['krylov']
    solver = KrylovSolver(sol, prec)

    # postproc / files
    ppdata = {}
    if param['io']['vtk'] or param['io']['timeseries'] or \
       param['io']['signal_TE']:
        utils.trymkdir(param['io']['results'])

    if param['io']['vtk']:
        vtkfile = File(param['io']['results'] + "/u_b" +
                       str(int(bval)).zfill(4) + "_.pvd")
        ppdata['vtkfile'] = vtkfile

    if param['io']['timeseries']:
        ppdata['Ulist'] = []
        ppdata['tlist'] = []

    if param['io']['timeseries'] or param['io']['signal_TE']:
        # create integration domain
        # append "dx_sig" Measure to meshdata dict
        meshdata = signal_domain(param, meshdata)

    param['bval'] = bval

    return meshdata, ppdata, solver


def initial_condition(V):
    ''' set initial condition and interpolate to function space V
        input:
            V:  function space
        return:
            ui: initial solution as function of V
    '''
    u0 = Constant((1.0, 0.0))
    ui = interpolate(u0, V)
    return ui


def abs_n(x):
    return 0.5*(x - abs(x))


def variational_problem(param, meshdata):
    ''' Get physics, define function spaces and weak form.
        Function spaces only live here.
        input:
            param:      dictionary with case setup
            meshdata:   dictionary with mesh, subdomains, boundaries
        return:
            ui:         initial solution
            (a_X,..):   tuple of linear and bilinear forms
    '''
    # ++++++   Physical properties   ++++++ #
    Gdir = param['phys']['pseq']['dir']
    dtp = param['phys']['pseq']['dt']
    k = param['phys']['D']
    bval = param['bval']

    # gyromagnetic ratio H1
    pi = DOLFIN_PI
    gm = 2*pi*42.57e3      # (Hz/mT)

    # compute needed gradient strength to achieve set b-value
    # assume Stejskal-Tanner PGSE rectangular pulses
    #  NOTE: this depends on pulse sequence -> def function!
    G = sqrt(bval/(gm**2*dtp[0]**2*(dtp[1]-dtp[0]/3.)))  # mT/mm

    # Gradient matrix operator, independent of time and pulse design
    Gr = Expression((('0', '-G*(x[0]*r0+x[1]*r1+x[2]*r2)'),
                     ('G*(x[0]*r0+x[1]*r1+x[2]*r2)', '0')),
                    G=G, r0=Gdir[0], r1=Gdir[1], r2=Gdir[2])

    # ++++++    Function space    +++++++ #
    V = FunctionSpace(meshdata['mesh'], "CG", param['num']['deg'])
    W = V*V
    dx = Measure('dx', domain=meshdata['mesh'], subdomain_data=meshdata['sd'])

    u = TrialFunction(W)
    v = TestFunction(W)

    uc = Constant(param['phys']['Uc'])
    sbf = Constant(param['phys']['backflow'])
    stemam = Constant(param['phys']['temam'])
    n = FacetNormal(meshdata['mesh'])

    # ++++++    Weak form    +++++++ #
    a_M = inner(u, v)*dx
    a_N = -gm*inner(dot(Gr, u), v)*dx
    a_C = (
        dot(grad(u)*uc, v)*dx -
        sbf*0.5*abs_n(dot(uc, n))*dot(u, v)*ds +
        stemam*0.5*div(uc)*dot(u, v)*dx
    )

    # diffusion: sum over all subdomains
    sdix = np.unique(meshdata['sd'].array())    # subdomain indices
    a_K = inner(Constant(k[0])*grad(u), grad(v))*dx(int(sdix[0]))
    for i in range(1, len(sdix)):
        a_K += inner(Constant(k[i])*grad(u), grad(v))*dx(i)

    # ++++++    Interpolate initial condition    +++++++ #
    ui = initial_condition(W)

    return ui, (a_M, a_N, a_K, a_C)


def preassemble(param, forms):
    ''' assemble time independent matrices and vectors
        input:
            param:  dictionary with case setup
            forms:  tuple with (bi)linear forms
        return:
            (A,..): tuple of pre-assembled matrices
    '''
    a_M, a_N, a_K, a_C = forms
    M = assemble(a_M)      # mass matrix
    N = assemble(a_N)      # reaction term
    K = assemble(a_K)      # diffusion
    C = assemble(a_C)      # convection

    # NOTE: Combining these matrices defines the time stepper.
    #       Generalize?
    dt = param['num']['dt']
    A0 = M + dt*(K + C)
    N *= dt

    return (A0, M, N)


def assemble_LS(param, mat, t, u):
    ''' assemble time dependent and non-linear terms
        input:
            param:  dictionary with case setup
            mat:    tuple of preassembled constant matrices
            u:      solution function
            t:      time
        return:
            A:  matrix of linear system
            b:  RHS of linear system
    '''
    A0, M, N = mat
    f = param['phys']['pseq']['f']
    dtp = param['phys']['pseq']['dt']

    A = A0 + f(t, dtp)*N
    b = M*u.vector()

    return A, b


def timestep(solver, A, b, u):
    ''' one day we will need this function! '''
    solver.solve(A, u.vector(), b)
    return u


def signal_domain(param, meshdata):
    # assume cube mesh
    class SignalDomain(SubDomain):
        def __init__(self, y1, y2):
            self.y1 = y1
            self.y2 = y2
            SubDomain.__init__(self)

        def inside(self, x, on_boundary):
            return (between(x[0], (self.y1, self.y2)) and
                    between(x[1], (self.y1, self.y2)) and
                    between(x[2], (self.y1, self.y2)))

    mesh = meshdata['mesh']
    # d = param['io']['signal_buffer']
    # xmin = np.min(mesh.coordinates())
    # xmax = np.max(mesh.coordinates())
    # L = xmax - xmin
    # y1 = xmin + L*d
    # y2 = xmax - L*d

    # absolute value (preferred)
    dx = param['io']['signal_domain']

    signaldomain = CellFunction('size_t', mesh)
    signaldomain.set_all(0)
    ssd = SignalDomain(-dx, dx)
    ssd.mark(signaldomain, 99)
    # meshdata['signaldomain'] = signaldomain
    meshdata['dx_sig'] = Measure('dx')[signaldomain]
    return meshdata


def postproc(param, ppdata, meshdata, u, t, T):
    ''' Compute derived quantities (signal), gather values,
        output to files and screen.
        input:
            param:  dictionary with case setup
            ppdata: dictionary with post processing settings
            u:      solution
            t:      time
            T:      end time
        return:
            U:      None or integral of u vector
            ppdata: updated post processing dict
    '''
    U = None
    # dt = param['num']['dt']
    io = param['io']
    # every = io['save_every']
    done = t >= T
    if MPI.rank(mpi_comm_world()) == 0:
        print(t)
    if not param['num']['dry']:  # and np.mod(t, dt*every) == 0:
        if io['vtk']:
            ppdata['vtkfile'] << u
        if io['timeseries']:
            # calculate U = integral(u)
            dxS = meshdata['dx_sig']
            U = [assemble(u[0]*dxS(99)), assemble(u[1]*dxS(99))]
            ppdata['Ulist'].append(U)
            ppdata['tlist'].append(t)

            if done and MPI.rank(mpi_comm_world()) == 0:
                import h5py
                fname = "b"+str(int(param['bval'])).zfill(4)+"_ts.h5"
                fdir = io['results']
                with h5py.File(fdir+"/"+fname, "w") as h5:
                    h5.create_dataset('time', data=ppdata['tlist'])
                    h5.create_dataset('U', data=ppdata['Ulist'])
                    h5.attrs['b'] = param['bval']
                    h5.close()

    if io['signal_TE'] and done:
        U = [assemble(u[0]*dxS(99)), assemble(u[1]*dxS(99))]

    return U, ppdata
