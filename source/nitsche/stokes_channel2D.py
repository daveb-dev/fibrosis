''' Channel flow with Navier-Slip (Robin type) boundary conditions
    author: David Nolte
    email: dnolte@dim.uchile.cl

    Two solvers are implemented:
        stokes2D:
            solve the Stokes equations on a rectangular mesh, with inlet on the
            left and outlet on the right border, and all other faces walls with
            BCs to be specified:
              - none: do nothing
              - no-slip (u=0 or horizontal u_wall) with Nitsche or DBC
              - free slip with Nitsche or DBC (horizontal wall)
              - navier-slip for inner section of a pipe, with gamma calculated
                from the known solution of a full Poiseuille flow (Nitsche)

        stokes2D_rotmesh:
            Same on a GMSH mesh rotated by "theta". The mesh's angle and radius
            need to be specified (todo: calculate?). What works:
              - none: do nothing
              - no-slip with u=0 (DBC and Nitsche)
              - free slip (Nitsche)
              - gamma-slip "inner tube" (Nitsche)
'''
from dolfin import *
# import sys
import numpy as np
# import matplotlib.pyplot as plt
# import scipy.linalg as la
# from sympy.utilities.codegen import ccode
from functions.geom import *


def stokes2D(mesh, beta=1000, solver='lu', gamma=0.0, prec='',
             poiseuille=False, wall='noslip', uwall=0.0, bctype='dirichlet',
             plots=False, periodic_inlet=False, inlet='const', Rref=None,
             umax=1.):
    ''' Stokes solver for 2D channel flow problems, using boundary markers
        defined in geom.py.
        input:
            mesh
            wall:       noslip, slip (free), navierslip (friction)
            uwall:      velocity of moving wall (noslip)
            bctype:     dirichlet, nitsche
            beta:       penalty for Nitsche BCs
            solver:     solver for linear system (e.g. mumps, lu or krylov
                        space method with suitable preconditioner)
            prec:       preconditioner for krylov space method
            poiseuille: bool, compare with exact poiseuille channel solution
            plots:      bool, plot solution
            periodic_inlet:  prepend periodic inlet zone to channel
            inlet:      inlet velocity profile, parabola or const
            Rref:       reference radius for inlet parabolic profile & bcs
            umax:       max. velocity for inflow
        output:
            u1, p1      solution
            W.dim()     number of DOFs
    '''

    ndim = mesh.geometry().dim()

    mu = 0.01

    k = Constant(mu)
    beta = Constant(beta)

    # Taylor-Hood Elements
    if periodic_inlet:
        pbc = PeriodicBoundaryINLET(0.2)
        V = VectorFunctionSpace(mesh, "CG", 2, constrained_domain=pbc)
        Q = FunctionSpace(mesh, "CG", 1, constrained_domain=pbc)
    else:
        V = VectorFunctionSpace(mesh, "CG", 2)
        Q = FunctionSpace(mesh, "CG", 1)

    W = V*Q

    coords = mesh.coordinates()
    xmin = coords[:, 0].min()
    xmax = coords[:, 0].max()
    # ymin = coords[:, 1].min()
    # ymax = coords[:, 1].max()
    left = Left(xmin)
    right = Right(xmax)
    # top = Top(ymin)
    # bottom = Bottom(ymax)
    allbnds = AllBoundaries()

    bnds = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    # ext mesh:
    # 1   inlet
    nin = 1
    # 2   outlet
    nout = 2
    # 3   wall
    ni = 3
    bnds.set_all(0)
    allbnds.mark(bnds, ni)
    # top.mark(bnds, ni)
    # bottom.mark(bnds, ni)
    left.mark(bnds, nin)
    right.mark(bnds, nout)

    ds = Measure('ds', domain=mesh, subdomain_data=bnds)

    zero = Constant((0,)*ndim)

    bcs = []
    # inflow Dirichlet BC
    if inlet == 'parabola':
        # y = a*(((xmax+xmin)/2 - x)**2 -(xmax-xmin)**2/4)
        if Rref:
            bmin = -Rref
            bmax = Rref
        else:
            # min/max coords on inlet boundary
            bmin = coords[coords[:, 0] == xmin, 1].min()
            bmax = coords[coords[:, 0] == xmin, 1].max()
        ys = (bmax + bmin) / 2.
        dy = (bmax - bmin) / 2.
        G = -umax/dy**2
        inflow = Expression(('G*(pow(ys - x[1], 2) - pow(dy, 2))', '0.0'),
                            ys=ys, dy=dy, G=G)
        # E = interpolate(inflow, V)
        # plot(E)
    elif inlet == 'const':
        inflow = Constant((umax, 0.0))

    bcin = DirichletBC(W.sub(0), inflow, bnds, 1)
    bcs.append(bcin)

    # outflow: du/dn + pn = 0 on ds(2)

    # Define variational problem
    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)

    n = FacetNormal(mesh)
    h = CellSize(mesh)

    F = zero

    def a(u, v):
        return inner(k*grad(u), grad(v))*dx

    def b(v, q):
        return div(v)*q*dx

    def c(u, v):
        return dot(u, dot(mu*grad(v), n))

    def f(v):
        return dot(F, v)*dx

    def t(u, p):
        return dot(k*grad(u), n) - p*n

    # alpha = Constant(1./10)

    # Standard form Stokes equations, Dirichlet RBs
    a1 = a(u, v) - b(v, p) - b(u, q)
    L1 = f(v)

    if bctype == 'dirichlet':
        bcin = DirichletBC(W.sub(0), inflow, bnds, nin)
        bcs.append(bcin)
    else:
        # stress tensor boundary integral
        a1 += - dot(t(u, p), v)*ds(nin)
        # Nitsche: u = inflow
        a1 += beta/h*k*dot(u, v)*ds(nin)
        L1 += beta/h*k*dot(inflow, v)*ds(nin)
        # consistent positive definite terms
        a1 += - dot(u, t(v, q))*ds(nin)
        L1 += - dot(inflow, t(v, q))*ds(nin)

    # wall treatment
    if wall == 'none':
        # "do nothing" boundary condition
        a1 += -dot(t(u, p), v)*ds(ni)

    if wall == 'navierslipSym':
        # normal component of standard boundary integral
        a1 += -(k*dot(dot(n, grad(u)), n) - p)*dot(v, n)*ds(ni)
        # tangential component replaced by Navier-Slip "gamma" BC
        #  (using  n.sigma.t = gamma u.t  and  v = v.n*n + v.t*t)
        a1 += -gamma*k*dot(u - dot(u, n)*n, v - dot(v, n)*n)*ds(ni)
        # Nitsche terms: weakly impose u.n = 0   (tg. comp. taken care of by
        #    Navier-Slip BC) and add skew-symmetric term
        a1 += beta/h*k*dot(u, n)*dot(v, n)*ds(ni) \
            - dot(u, n)*(k*dot(dot(n, grad(v)), n) - q)*ds(ni)

    if wall == 'slip' or wall == 'navierslip':
        # u.n = 0, t-(t.n)n = 0
        # normal component is 0 anyways for Dirichlet BC on u_1 !!

        if bctype == 'dirichlet':
            # normal component == 0, works only if horizontal wall, n=ey
            bc = DirichletBC(W.sub(0).sub(1), 0.0, bnds, ni)
            bcs.append(bc)

        elif bctype == 'nitsche':
            # impose u.n = 0   (=> add "nothing" on rhs)
            a1 += beta/h*dot(u, n)*dot(v, n)*ds(ni)

        if wall == 'slip':
            # THIS is wrong, needs to be reverse!
            #   -> tangential comp = 0, t.n stays!
            # a1 += -dot(t(u, p) - dot(t(u, p), n)*n, v - dot(v, n)*n)*ds(ni)
            a1 += -dot(t(u, p), n)*dot(v, n)*ds(ni)

        if wall == 'navierslip':
            # normal comp of stress vector (=0 if DBC) + tangential part via
            #  Robin BC
            a1 += -dot(t(u, p), n)*dot(v, n)*ds(ni) - gamma*k*dot(u, v)*ds(ni)

    #
    if wall == 'noslip':
        noslip = Constant((uwall, 0.0))
        if bctype == 'dirichlet':
            bc0 = DirichletBC(W.sub(0), noslip, bnds, ni)
            bcs.append(bc0)
        elif bctype == 'nitsche':
            # why add k=mu ?
            a1 += -dot(t(u, p), v)*ds(ni) + beta/h*k*dot(u, v)*ds(ni)
            L1 += beta/h*k*dot(noslip, v)*ds(ni)

    if wall == 'noslipSym':
        # from stokes_cyl3d:
        noslip = Constant((uwall, 0.0))
        a1 += - dot(t(u, p), v)*ds(ni)
        a1 += beta/h*k*dot(u, v)*ds(ni)
        L1 += beta/h*k*dot(noslip, v)*ds(ni)
        # consistent positive definite terms
        a1 += - dot(u, t(v, q))*ds(ni)
        L1 += - dot(noslip, t(v, q))*ds(ni)

    # Solve
    w1 = Function(W)
    solve(a1 == L1, w1, bcs, solver_parameters={'linear_solver': solver,
                                                'preconditioner': prec})

    # Split the mixed solution
    (u1, p1) = w1.split(deepcopy=True)

    print("Norm of velocity coefficient vector: %.6g" % u1.vector().norm("l2"))
    print("Norm of pressure coefficient vector: %.6g" % p1.vector().norm("l2"))

    # err_u = 0
    # if poiseuille:
    #     err_u = errornorm(inflow, u1)
    #     print('Error norm wrt Poiseuille solution U:  D %.6g' % err_u)

    if plots:
        plot(u1, title='velocity, dirichlet')
        plot(p1, title='pressure, dirichlet')

    return u1, p1, W.dim()


def stokes2D_rotmesh(meshfile, beta=1000, solver='lu', gamma=0.0, prec='',
                     poiseuille=False, wall='noslip', uwall=0.0,
                     bctype='dirichlet', plots=False, periodic_inlet=False,
                     inlet='const', Rref=0.5, theta=0., umax=1.):
    ''' Stokes solver for 2D channel flow problems, using GMSH boundary
        indicators for ROTATED MESH.
        SPECIFY Rref and theta according to GEOMETRY.

        EXPECTED BCs TO WORK:
            Nitsche: noslip (u=0), navierslip, (slip)
            Dirichlet: noslip (u=0)

        input:
            mesh
            wall:       noslip, slip (free), navierslip (friction)
            uwall:      velocity of moving wall (noslip)
            bctype:     dirichlet, nitsche
            beta:       penalty for Nitsche BCs
            solver:     solver for linear system (e.g. mumps, lu or krylov
                        space method with suitable preconditioner)
            prec:       preconditioner for krylov space method
            poiseuille: bool, compare with exact poiseuille channel solution
            plots:      bool, plot solution
            periodic_inlet:  prepend periodic inlet zone to channel
            inlet:      inlet velocity profile, parabola or const
            Rref:       reference radius for inlet parabolic profile & bcs
            umax:       max. velocity for inflow
        output:
            u1, p1      solution
            W.dim()     number of DOFs
    '''

    from functions.inout import read_mesh

    mesh, _, bnds = read_mesh(meshfile)
    # plot(bnds)

    ndim = mesh.geometry().dim()

    mu = 0.01

    k = Constant(mu)
    beta = Constant(beta)

    # Taylor-Hood Elements
    if periodic_inlet:
        pbc = PeriodicBoundaryINLET(0.2)
        V = VectorFunctionSpace(mesh, "CG", 2, constrained_domain=pbc)
        Q = FunctionSpace(mesh, "CG", 1, constrained_domain=pbc)
    else:
        V = VectorFunctionSpace(mesh, "CG", 2)
        Q = FunctionSpace(mesh, "CG", 1)

    W = V*Q

    # TODO: GET CORRECT COORDINATES FOR ROTATED PARABOLA

    # allbnds = AllBoundaries()

    # bnds = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    # ni = 3
    # bnds.set_all(0)
    # allbnds.mark(bnds, ni)
    # # top.mark(bnds, ni)
    # # bottom.mark(bnds, ni)
    # left.mark(bnds, 1)
    # right.mark(bnds, 2)

    # Physical group WALL
    ni = 3
    # PG Inlet
    nin = 1
    # PG Outlet (not needed)
    nout = 2

    ds = Measure('ds', domain=mesh, subdomain_data=bnds)

    zero = Constant((0,)*ndim)

    bcs = []
    # inflow Dirichlet BC
    if inlet == 'parabola':
        # y = a*(((xmax+xmin)/2 - x)**2 -(xmax-xmin)**2/4)

        # from dolfin import *
        # from functions.inout import read_mesh
        # mesh, _, _ = read_mesh('meshes/chan2D_rot.h5')

        # Rref = 0.5
        # umax = 2.
        G = -umax/Rref**2
        # theta = -DOLFIN_PI/4.

        inflow = Expression((
            'G*cos(th)*(pow(sin(th)*x[0] + cos(th)*x[1], 2) - R*R)',
            '-G*sin(th)*(pow(sin(th)*x[0] + cos(th)*x[1], 2) - R*R)'),
            R=Rref, G=G, th=theta)

        # V = VectorFunctionSpace(mesh, "CG", 1)
        # U = interpolate(inflow, V)
        # plot(U, interactive=True)

        # inflow = Expression(('G*(pow(ys - x[1], 2) - pow(dy, 2))', '0.0'),
        #                     ys=ys, dy=dy, G=G, th=theta)
        # E = interpolate(inflow, V)
        # plot(E)
    elif inlet == 'const':
        inflow = Constant((umax*cos(th), -umax*sin(th)))  # XXX check this?


    # outflow: du/dn + pn = 0 on ds(2)

    # Define variational problem
    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)

    n = FacetNormal(mesh)
    h = CellSize(mesh)

    F = zero

    def a(u, v):
        return inner(k*grad(u), grad(v))*dx

    def b(v, q):
        return div(v)*q*dx

    def c(u, v):
        return dot(u, dot(mu*grad(v), n))

    def f(v):
        return dot(F, v)*dx

    def t(u, p):
        return dot(k*grad(u), n) - p*n

    # alpha = Constant(1./10)

    # Standard form Stokes equations
    a1 = a(u, v) - b(v, p) - b(u, q)
    L1 = f(v)

    if bctype == 'dirichlet':
        bcin = DirichletBC(W.sub(0), inflow, bnds, nin)
        bcs.append(bcin)
    else:
        # stress tensor boundary integral
        a1 += - dot(t(u, p), v)*ds(nin)
        # Nitsche: u = inflow
        a1 += beta/h*k*dot(u, v)*ds(nin)
        L1 += beta/h*k*dot(inflow, v)*ds(nin)
        # consistent positive definite terms
        a1 += - dot(u, t(v, q))*ds(nin)
        L1 += - dot(inflow, t(v, q))*ds(nin)

    # wall treatment
    if wall == 'none':
        # "do nothing" boundary condition
        a1 += -dot(t(u, p), v)*ds(ni)

    if wall == 'navierslipSym':
        # normal component of standard boundary integral
        a1 += -(k*dot(dot(n, grad(u)), n) - p)*dot(v, n)*ds(ni)
        # tangential component replaced by Navier-Slip "gamma" BC
        #  (using  n.sigma.t = gamma u.t  and  v = v.n*n + v.t*t)
        a1 += -gamma*k*dot(u - dot(u, n)*n, v - dot(v, n)*n)*ds(ni)
        # Nitsche terms: weakly impose u.n = 0   (tg. comp. taken care of by
        #    Navier-Slip BC) and add skew-symmetric term
        a1 += beta/h*k*dot(u, n)*dot(v, n)*ds(ni) \
            - dot(u, n)*(k*dot(dot(n, grad(v)), n) - q)*ds(ni)

    if wall == 'slip' or wall == 'navierslip':
        # u.n = 0, t-(t.n)n = 0
        # normal component is 0 anyways for Dirichlet BC on u_1 !!

        if bctype == 'dirichlet':
            # normal component == 0, works only if horizontal wall, n=ey
            print('WARNING: u.n via DBC works only for HORIZONTAL WALL!')
            bc = DirichletBC(W.sub(0).sub(1), 0.0, bnds, ni)
            bcs.append(bc)

        elif bctype == 'nitsche':
            # impose u.n = 0   (=> add "nothing" on rhs)
            a1 += beta/h*dot(u, n)*dot(v, n)*ds(ni)

        if wall == 'slip':
            # THIS is wrong, needs to be reverse!
            #   -> tangential comp = 0, t.n stays!
            # a1 += -dot(t(u, p) - dot(t(u, p), n)*n, v - dot(v, n)*n)*ds(ni)
            a1 += -dot(t(u, p), n)*dot(v, n)*ds(ni)

        if wall == 'navierslip':
            # normal comp of stress vector (=0 if DBC) + tangential part via
            #  Robin BC
            a1 += -dot(t(u, p), n)*dot(v, n)*ds(ni) - gamma*k*dot(u, v)*ds(ni)

    #
    uw_x = uwall*np.cos(theta)
    uw_y = -uwall*np.sin(theta)
    noslip = Constant((uw_x, uw_y))
    if wall == 'noslip':
        if bctype == 'dirichlet':
            bc0 = DirichletBC(W.sub(0), noslip, bnds, ni)
            bcs.append(bc0)
        elif bctype == 'nitsche':
            # why add k=mu ?
            a1 += -dot(t(u, p), v)*ds(ni) + beta/h*k*dot(u, v)*ds(ni)
            L1 += beta/h*k*dot(noslip, v)*ds(ni)

    if wall == 'noslipSym':
        # from stokes_cyl3d:
        noslip = Constant((uw_x, uw_y))
        a1 += - dot(t(u, p), v)*ds(ni)
        a1 += beta/h*k*dot(u, v)*ds(ni)
        L1 += beta/h*k*dot(noslip, v)*ds(ni)
        # consistent positive definite terms
        a1 += - dot(u, t(v, q))*ds(ni)
        L1 += - dot(noslip, t(v, q))*ds(ni)

    # Solve
    w1 = Function(W)
    solve(a1 == L1, w1, bcs, solver_parameters={'linear_solver': solver,
                                                'preconditioner': prec})

    # Split the mixed solution
    (u1, p1) = w1.split(deepcopy=True)

    print("Norm of velocity coefficient vector: %.6g" % u1.vector().norm("l2"))
    print("Norm of pressure coefficient vector: %.6g" % p1.vector().norm("l2"))

    # err_u = 0
    # if poiseuille:
    #     err_u = errornorm(inflow, u1)
    #     print('Error norm wrt Poiseuille solution U:  D %.6g' % err_u)

    if plots:
        plot(u1, title='velocity, dirichlet')
        plot(p1, title='pressure, dirichlet')

        ue = interpolate(inflow, V)
        plot((u1 - ue))

    return u1, p1, W.dim()


def narrow_vs_full_tube():
    N = 2**3
    Ri = 1.0
    R = 1.1
    L = 4.
    umax = 3.
    gamma = 2.*Ri/(Ri**2 - R**2)
    beta = 20.

    def uwall(umax, R, y):
        return -umax/R**2*(y**2 - R**2)

    mesh1 = tube_mesh2D(Ri, L, N)
    u1, p1, _ = stokes2D(mesh1,
                         wall='navierslipSym',
                         uwall=uwall(umax, R, Ri),  # noslip only
                         bctype='dirichlet',
                         gamma=gamma,            # navier-slip only
                         beta=beta,              # Nitsche only
                         solver='mumps',
                         plots=True,
                         periodic_inlet=False,
                         inlet='parabola',  # parabola,const
                         Rref=R,
                         umax=umax
                         )
    # DONE: gives exact solution for all noslip(Sym)/navierslip(Sym) D/N
    #  combinations

    # compare with Poiseuille flow
    # ys = (bmax + bmin) / 2.
    # dy = (bmax - bmin) / 2.
    # V = VectorFunctionSpace(mesh1, 'Lagrange', 2)
    ys = 0
    dy = R
    G = -umax/dy**2
    inflow = Expression(('G*(pow(ys-x[1], 2) - pow(dy, 2))', '0.0'), ys=ys,
                        dy=dy, G=G)
    # uexact = interpolate(inflow, V)

    err = errornorm(inflow, u1)
    print("Error norm %f" % err)

def check_rotated():
    # XXX possible BC combinations:
    #   DBC + noslip = 0
    #   Nitsche + noslip(0), slip, navierslip
    # N = 2**7
    Ri = 0.5
    R = 0.6
    # L = 5.
    umax = 2.
    gamma = 2.*Ri/(Ri**2 - R**2)
    beta = 1000.

    theta = -0.7853981

    def uwall(umax, R, y):
        return -umax/R**2*(y**2 - R**2)

    # mesh1 = tube_mesh2D(Ri, L, N)
    mesh1 = './meshes/chan2D_rot.h5'
    u1, p1, _ = stokes2D_rotmesh(
        mesh1,
        wall='noslipSym',
        uwall=uwall(umax, R, Ri),  # noslip only
        # uwall=0.0,
        bctype='nitsche',
        gamma=gamma,            # navier-slip only
        beta=beta,              # Nitsche only
        solver='mumps',
        plots=True,
        periodic_inlet=False,
        inlet='parabola',  # parabola,const
        Rref=R,
        theta=theta,
        umax=umax
    )
    # DONE: gives exact solution for all noslip(Sym)/navierslip(Sym) D/N
    #  combinations

    # compare with Poiseuille flow
    # ys = (bmax + bmin) / 2.
    # dy = (bmax - bmin) / 2.
    # V = VectorFunctionSpace(mesh1, 'Lagrange', 2)
    G = -umax/R**2
    inflow = Expression((
        'G*cos(th)*(pow(sin(th)*x[0] + cos(th)*x[1], 2) - R*R)',
        '-G*sin(th)*(pow(sin(th)*x[0] + cos(th)*x[1], 2) - R*R)'),
        R=R, G=G, th=theta)
    # uexact = interpolate(inflow, V)

    err = errornorm(inflow, u1)
    print("Error norm %.10f" % err)


if __name__ == '__main__':
    # narrow_vs_full_tube()
    check_rotated()
