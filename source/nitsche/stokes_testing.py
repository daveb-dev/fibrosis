''' Nitsche method applied on the Stokes equation
    cf. Freund, Stenberg (1995): "On weakly imposed boundary conditions for
    second order problems"
'''
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
# import scipy.linalg as la
# from sympy.utilities.codegen import ccode
from functions.geom import *


def stokes2D_slip(mesh, beta=1000, solver='lu', prec='', poiseuille=False,
                  plots=False, periodic_inlet=False, inlet='const', umax=1):

    # XXX: TOP = SLIP BOUNDARY
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

    W = V * Q

    coords = mesh.coordinates()
    xmin = coords[:, 0].min()
    xmax = coords[:, 0].max()
    # ymin = coords[:, 1].min()
    ymax = coords[:, 1].max()
    left = Left(xmin)
    right = Right(xmax)
    top = Top(ymax)
    # bottom = Bottom(ymin)
    allbnds = AllBoundaries()

    bnds = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    ni = 3
    ni2 = 4
    bnds.set_all(0)
    allbnds.mark(bnds, ni)
    top.mark(bnds, ni2)
    # bottom.mark(bnds, ni)
    left.mark(bnds, 1)
    right.mark(bnds, 2)

    ds = Measure('ds', domain=mesh, subdomain_data=bnds)

    zero = Constant((0,) * ndim)
    noslip = zero
    if inlet == 'parabola':
        # y = a*(((xmax+xmin)/2 - x)**2 -(xmax-xmin)**2/4)
        # min/max coords on inlet boundary
        bmin = coords[coords[:, 0] == xmin, 1].min()
        bmax = coords[coords[:, 0] == xmin, 1].max()
        ys = (bmax + bmin) / 2.
        dy = (bmax - bmin) / 2.
        G = -umax/dy**2
        inflow = Expression(('G*(pow(ys-x[1], 2) - pow(dy,2))', '0.0'),
                            ys=ys, dy=dy, G=G)
    elif inlet == 'const':
        inflow = Constant((umax, 0.0))
    # outflow: du/dn + pn = 0 on ds(2)

    bc0 = DirichletBC(W.sub(0), noslip, bnds, ni)
    bc1 = DirichletBC(W.sub(0), inflow, bnds, 1)
    # u.n = 0   &&  s(u,p).n.t = 0 (Neumann)
    bc2 = DirichletBC(W.sub(0).sub(1), 0.0, bnds, ni2)
    bcs = [bc0, bc1, bc2]

    F = zero
    # Define variational problem
    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)

    n = FacetNormal(mesh)
    h = CellSize(mesh)

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
    a1 = a(u, v) - b(v, p) + b(u, q) - dot(t(u, p), n)*dot(v, n)*ds(ni2)
    L1 = f(v)

    w1 = Function(W)
    solve(a1 == L1, w1, bcs, solver_parameters={'linear_solver': solver,
                                                'preconditioner': prec})

    # Nitsche's standard method, impose NOSLIP BC, without SUPG/stability terms
    # XXX: +b(u, q) missing in Ref!

    # a2 = a1 - dot(t(u, p), v)*ds(ni) - \
    #         c(u, v)*ds(ni) +  beta/h*dot(u, k*v)*ds(ni)

    # or: (DIFFERENCE: use mu*grad(v) or t(v,q)*n)
    a2 = a1 - dot(t(u, p), v)*ds(ni) - \
        dot(t(u, p), n)*dot(v, n)*ds(ni2) + \
        beta/h*dot(u, n)*dot(v, n)*ds(ni2) - \
        dot(u, t(v, q))*ds(ni) + beta/h*dot(u, k*v)*ds(ni)

    # SUPG term:
    #     - alpha*h**2*dot(grad(p), grad(q))*dx

    # L2 = f(v) - c(noslip, v)*ds(ni) + beta/h*dot(noslip, k*v)*ds(ni)
    L2 = f(v)
    # SUPG term:
    #         + alpha*h**2*dot(F, grad(q))*dx
    # L  == f(v) == 0
    # L2 = f(v) - dot(noslip, t(v, q))*ds(ni) + beta/h*dot(noslip, k*v)*ds(ni)

    w2 = Function(W)
    solve(a2 == L2, w2, bc1, solver_parameters={'linear_solver': solver,
                                                'preconditioner': prec})

    # Split the mixed solution using a shallow copy
    (u1, p1) = w1.split(deepcopy=True)
    (u2, p2) = w2.split(deepcopy=True)

    print("DIRICHLET SLIP")
    print("Norm of velocity coefficient vector: %.6g" % u1.vector().norm("l2"))
    print("Norm of pressure coefficient vector: %.6g" % p1.vector().norm("l2"))
    print("NITSCHE SLIP")
    print("Norm of velocity coefficient vector: %.6g" % u2.vector().norm("l2"))
    print("Norm of pressure coefficient vector: %.6g" % p2.vector().norm("l2"))

    err_u = (u1.vector() - u2.vector()).norm("l2")/u1.vector().norm("l2")
    if poiseuille:
        # pe = Expression('G*(L/2-x[1])', G=G, L=Ly)
        err_u = [errornorm(inflow, u1), errornorm(inflow, u2)]
        print('Error norm wrt Poiseuille solution U:  D %.6g \t N %.6g' %
              (err_u[0], err_u[1]))
        # print('Error norm wrt Poiseuille solution P:  %.6g' % err_p)

    if plots:
        plot(u1, title='velocity, dirichlet')
        plot(p1, title='pressure, dirichlet')
        plot(u2, title='velocity, nitsche')
        plot(p2, title='pressure, nitsche')
        # interactive()

    return (u1, p1), (u2, p2), err_u, W.dim()


def stokes2D_robin(mesh, gamma, beta=1000, solver='lu', prec='',
                   poiseuille=False, plots=False, periodic_inlet=False,
                   inlet='const', Rfull=0, umax=1):

    # XXX: TOP = SLIP BOUNDARY
    ndim = mesh.geometry().dim()

    mu = 0.01

    k = Constant(mu)
    beta = Constant(beta)
    gamma = Constant(gamma)

    # Taylor-Hood Elements
    if periodic_inlet:
        pbc = PeriodicBoundaryINLET(1.)
        V = VectorFunctionSpace(mesh, "CG", 2, constrained_domain=pbc)
        Q = FunctionSpace(mesh, "CG", 1, constrained_domain=pbc)
    else:
        V = VectorFunctionSpace(mesh, "CG", 2)
        Q = FunctionSpace(mesh, "CG", 1)

    W = V * Q

    coords = mesh.coordinates()
    xmin = coords[:, 0].min()
    xmax = coords[:, 0].max()
    ymin = coords[:, 1].min()
    ymax = coords[:, 1].max()
    left = Left(xmin)
    right = Right(xmax)
    top = Top(ymax)
    bottom = Bottom(ymin)
    # allbnds = AllBoundaries()

    bnds = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    ni = 3
    bnds.set_all(0)
    # allbnds.mark(bnds, ni)
    left.mark(bnds, 1)
    right.mark(bnds, 2)
    top.mark(bnds, ni)
    bottom.mark(bnds, ni)

    ds = Measure('ds', domain=mesh, subdomain_data=bnds)

    zero = Constant((0,) * ndim)
    # noslip = zero
    if inlet == 'parabola':
        # y = a*(((xmax+xmin)/2 - x)**2 -(xmax-xmin)**2/4)
        if Rfull:
            bmin = -Rfull
            bmax = Rfull
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

    # outflow: du/dn + pn = 0 on ds(2)

    # bc0 = DirichletBC(W.sub(0), noslip, bnds, ni)
    bc1 = DirichletBC(W.sub(0), inflow, bnds, 1)
    # u.n = 0   &&  s(u,p).n.t = 0 (Neumann)
    bc2 = DirichletBC(W.sub(0).sub(1), 0.0, bnds, ni)
    bcs = [bc1, bc2]

    F = zero
    # Define variational problem
    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)

    n = FacetNormal(mesh)
    h = CellSize(mesh)

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
        # return dot(k*2*sym(grad(u)), n) - p*n

    # alpha = Constant(1./10)

    # Stokes variational form with Navier-Slip BCs on ds(ni)
    #   Int T.v ds = Int (T.n)(v.n) - gm*u.v + gm*(u.n)(v.n) ds
    #   let u.n = 0 implicitly: ommit last term
    # TODO: check if setting u.n via Dirichlet or Nitsche changes the result
    print("gamma = %f" % gamma)
    a1 = a(u, v) - b(v, p) + b(u, q) - \
        (dot(n, t(u, p))*dot(v, n))*ds(ni) + gamma*dot(u, v)*ds(ni)
    # a1 += gamma*dot(u, n)*dot(v, n)*ds(ni)    # u.n = 0!
    L1 = f(v)

    w1 = Function(W)
    solve(a1 == L1, w1, bcs, solver_parameters={'linear_solver': solver,
                                                'preconditioner': prec})

    # # Nitsche's standard method, impose NOSLIP BC, without SUPG/stab terms
    # # XXX: +b(u, q) missing in Ref!

    # # a2 = a1 - dot(t(u, p), v)*ds(ni) - \
    # #         c(u, v)*ds(ni) +  beta/h*dot(u, k*v)*ds(ni)

    # # or: (DIFFERENCE: use mu*grad(v) or t(v,q)*n)
    # a2 = a1 - dot(t(u, p), v)*ds(ni) - \
    #     dot(t(u, p), n)*dot(v, n)*ds(ni2) + \
    #     beta/h*dot(u, n)*dot(v, n)*ds(ni2) - \
    #     dot(u, t(v, q))*ds(ni) + beta/h*dot(u, k*v)*ds(ni)

    # # SUPG term:
    # #     - alpha*h**2*dot(grad(p), grad(q))*dx

    # # L2 = f(v) - c(noslip, v)*ds(ni) + beta/h*dot(noslip, k*v)*ds(ni)
    # L2 = f(v)
    # # SUPG term:
    # #         + alpha*h**2*dot(F, grad(q))*dx
    # # L  == f(v) == 0
    # # L2 = f(v) - dot(noslip, t(v, q))*ds(ni) + beta/h*dot(noslip, k*v)*ds(ni)

    # w2 = Function(W)
    # solve(a2 == L2, w2, bc1, solver_parameters={'linear_solver': solver,
    #                                             'preconditioner': prec})

    # Split the mixed solution using a shallow copy
    (u1, p1) = w1.split(deepcopy=True)
    # (u2, p2) = w2.split(deepcopy=True)

    print("Navier-slip via ROBIN BC")
    print("Norm of velocity coefficient vector: %.6g" % u1.vector().norm("l2"))
    print("Norm of pressure coefficient vector: %.6g" % p1.vector().norm("l2"))
    # print("NITSCHE SLIP")
    # print("Norm of velocity coefficient vector: %.6g" % u2.vector().norm("l2"))
    # print("Norm of pressure coefficient vector: %.6g" % p2.vector().norm("l2"))

    # err_u = (u1.vector() - u2.vector()).norm("l2")/u1.vector().norm("l2")
    # if poiseuille:
    #     # pe = Expression('G*(L/2-x[1])', G=G, L=Ly)
    #     err_u = [errornorm(inflow, u1), errornorm(inflow, u2)]
    #     print('Error norm wrt Poiseuille solution U:  D %.6g \t N %.6g' %
    #           (err_u[0], err_u[1]))
    #     # print('Error norm wrt Poiseuille solution P:  %.6g' % err_p)

    if plots:
        plot(u1, title='velocity, Robin BC')
        plot(p1, title='pressure, dirichlet')
        # plot(u2, title='velocity, nitsche')
        # plot(p2, title='pressure, nitsche')
        # interactive()

    return u1, p1  # , (u2, p2), err_u, W.dim()


def stokes2D(mesh, beta=1000, solver='lu', prec='', poiseuille=False,
             plots=False, periodic_inlet=False, inlet='const', umax=1):

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
    ni = 3
    bnds.set_all(0)
    allbnds.mark(bnds, ni)
    # top.mark(bnds, ni)
    # bottom.mark(bnds, ni)
    left.mark(bnds, 1)
    right.mark(bnds, 2)

    ds = Measure('ds', domain=mesh, subdomain_data=bnds)

    zero = Constant((0,)*ndim)
    noslip = zero
    if inlet == 'parabola':
        # y = a*(((xmax+xmin)/2 - x)**2 -(xmax-xmin)**2/4)
        # min/max coords on inlet boundary
        bmin = coords[coords[:, 0] == xmin, 1].min()
        bmax = coords[coords[:, 0] == xmin, 1].max()
        ys = (bmax + bmin)/2.
        dy = (bmax - bmin)/2.
        G = -umax/dy**2
        inflow = Expression(('G*(pow(ys - x[1], 2) - pow(dy, 2))', '0.0'),
                            ys=ys, dy=dy, G=G)
    elif inlet == 'const':
        inflow = Constant((umax, 0.0))
    # outflow: du/dn + pn = 0 on ds(2)

    bc0 = DirichletBC(W.sub(0), noslip, bnds, ni)
    bc1 = DirichletBC(W.sub(0), inflow, bnds, 1)
    bcs = [bc0, bc1]

    F = zero
    # Define variational problem
    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)

    n = FacetNormal(mesh)
    h = CellSize(mesh)

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
    a1 = a(u, v) - b(v, p) + b(u, q)
    L1 = f(v)

    w1 = Function(W)
    solve(a1 == L1, w1, bcs, solver_parameters={'linear_solver': solver,
                                                'preconditioner': prec})

    # Nitsche's standard method, impose NOSLIP BC, without SUPG/stability terms
    # XXX: +b(u, q) missing in Ref!

    # a2 = a1 - dot(t(u, p), v)*ds(ni) - \
    #         c(u, v)*ds(ni) +  beta/h*dot(u, k*v)*ds(ni)

    # or: (DIFFERENCE: use mu*grad(v) or t(v,q)*n)
    a2 = a1 - dot(t(u, p), v)*ds(ni) - \
        dot(u, t(v, q))*ds(ni) + beta/h*dot(u, k*v)*ds(ni)

    # SUPG term:
    #     - alpha*h**2*dot(grad(p), grad(q))*dx

    # L2 = f(v) - c(noslip, v)*ds(ni) + beta/h*dot(noslip, k*v)*ds(ni)
    L2 = f(v)
    # SUPG term:
    #         + alpha*h**2*dot(F, grad(q))*dx
    # L  == f(v) == 0
    # L2 = f(v) - dot(noslip, t(v, q))*ds(ni) + beta/h*dot(noslip, k*v)*ds(ni)

    w2 = Function(W)
    solve(a2 == L2, w2, bc1, solver_parameters={'linear_solver': solver,
                                                'preconditioner': prec})

    # Split the mixed solution using a shallow copy
    (u1, p1) = w1.split(deepcopy=True)
    (u2, p2) = w2.split(deepcopy=True)

    print("DIRICHLET")
    print("Norm of velocity coefficient vector: %.6g" % u1.vector().norm("l2"))
    print("Norm of pressure coefficient vector: %.6g" % p1.vector().norm("l2"))
    print("NITSCHE")
    print("Norm of velocity coefficient vector: %.6g" % u2.vector().norm("l2"))
    print("Norm of pressure coefficient vector: %.6g" % p2.vector().norm("l2"))

    err_u = (u1.vector() - u2.vector()).norm("l2")/u1.vector().norm("l2")
    if poiseuille:
        # pe = Expression('G*(L/2-x[1])', G=G, L=Ly)
        err_u = [errornorm(inflow, u1), errornorm(inflow, u2)]
        print('Error norm wrt Poiseuille solution U:  D %.6g \t N %.6g' %
              (err_u[0], err_u[1]))
        # print('Error norm wrt Poiseuille solution P:  %.6g' % err_p)

    if plots:
        plot(u1, title='velocity, dirichlet')
        plot(p1, title='pressure, dirichlet')
        plot(u2, title='velocity, nitsche')
        plot(p2, title='pressure, nitsche')
        # interactive()

    return (u1, p1), (u2, p2), err_u, W.dim()


def DBCvsNitsche_poiseuille():
    R = 1.0
    Ly = 5.0
    err = []
    dofs = []
    # h loop:
    p = range(4, 8)
    beta = 1000
    for q in p:
        mesh = tube_mesh2D(R*2, Ly, 2**q)
        _, _, tmp, dof = stokes2D(mesh,
                                  poiseuille=True,
                                  plots=False,
                                  beta=beta,
                                  solver='mumps',
                                  prec='',
                                  periodic_inlet=True,
                                  inlet='const'   # parabola
                                  )
        err.append(tmp)
        dofs.append(dof)

    E = np.array(err)
    print('beta = %d' % beta)
    print('DOFs\t|\tDirichlet ||u-wi||\t|\tNitsche ||u-wi||')
    for i, dof in enumerate(dofs):
        print('%i\t%.16f\t%.16f' % (dof, err[i][0], err[i][1]))
    plt.figure()
    plt.loglog(dofs, E, '-o')
    plt.title('Stokes 2D channel flow (Poiseuille)')
    plt.xlabel('DOFs')
    plt.ylabel(r'$|||e|||$')
    plt.legend(('Dirichlet', 'Nitsche'), loc='lower left')
    plt.axis('scaled')

    # beta loop:
    p = 7
    bexp = np.arange(-1, 6)
    beta = 10.**bexp

    err = []
    for x in beta:
        mesh = tube_mesh2D(R*2, Ly, 2**p)
        _, _, tmp, dof = stokes2D(mesh,
                                  poiseuille=True,
                                  plots=False,
                                  beta=x,
                                  solver='mumps',
                                  prec=''
                                  )
        err.append(tmp)

    E = np.array(err)
    print('DOFs = %d' % dof)
    print('beta\t|\tDirichlet ||u-wi||\t|\tNitsche ||u-wi||')
    for i, x in enumerate(beta):
        print('%i\t%.16f\t%.16f' % (x, err[i][0], err[i][1]))
    plt.figure()
    plt.loglog(beta, E, '-o')
    plt.title('Stokes 2D channel flow (Poiseuille)')
    plt.xlabel(r'\beta')
    plt.ylabel(r'$|||e|||$')
    plt.legend(('Dirichlet', 'Nitsche'), loc='lower left')
    plt.axis('scaled')


def DBCvsNitsche_bench():
    N = 2**7
    beta = 1000
    mesh1 = BFSmesh(1, 0.2, 2, 2, N)
    mesh2 = cylinderflow_mesh(1.5, 5, 1, 0.2, N)
    meshes = [mesh1, mesh2]
    problems = [stokes2D_slip]      # [stokes2D, stokes2D_slip]
    k = 0
    for solver in problems:
        i = 0
        for mesh in meshes:
            (u1, p1), (u2, p2), _, _ = \
                solver(mesh,
                       beta=beta,
                       solver='mumps',
                       poiseuille=False,
                       plots=True,
                       periodic_inlet=True,
                       inlet='parabola',  # parabola,const
                       umax=3
                       )
            file1 = File("results/dirichlet_%s_%s.pvd" % (mesh._name,
                                                          solver.__name__))
            file1 << u1, p1
            file2 = File("results/nitsche_%s_%s.pvd" % (mesh._name,
                                                        solver.__name__))
            file2 << u2, p2
            i += 1
        k += 1


def narrow_vs_full_tube():
    N = 2**7
    Ri = 0.8
    R = 1.0
    L = 5.
    umax = 5.
    gamma = -2.*Ri/(Ri**2 - R**2)
    # *(R-Ri)  # more exact if multiplied by (R-Ri)! WHY?
    mesh1 = tube_mesh2D(Ri, L, N)
    u1, p1 = stokes2D_robin(mesh1,
                            gamma=gamma,
                            solver='mumps',
                            poiseuille=False,
                            plots=True,
                            periodic_inlet=False,
                            inlet='parabola',  # parabola,const
                            Rfull=R,
                            umax=umax
                            )

    # compare with Poiseuille flow
    # ys = (bmax + bmin) / 2.
    # dy = (bmax - bmin) / 2.
    V = VectorFunctionSpace(mesh1, 'Lagrange', 2)
    ys = 0
    dy = R
    G = -umax/dy**2
    inflow = Expression(('G*(pow(ys-x[1], 2) - pow(dy,2))', '0.0'), ys=ys,
                        dy=dy, G=G)
    uexact = interpolate(inflow, V)

    err = errornorm(inflow, u1)
    print("Error norm %f" % err)

    # plot(uexact, title="analytic")
    # plot(uexact - u1, title="difference")

    # file1 = File("results/robin0.pvd")
    # file1 << u1
    # file2 = File("results/poiseuille.pvd")
    # file2 << uexact

    # mesh2 = tube_mesh2D(R, L, N)
    # (u2, p2), _, _, _, = stokes2D(mesh2,
    #                               solver='mumps',
    #                               poiseuille=False,
    #                               plots=True,
    #                               periodic_inlet=False,
    #                               inlet='parabola',  # parabola,const
    #                               umax=3
    #                               )

    # TODO: split stokes2D() in stokes2D_dirichlet and stokes2D_nitsche and
    #   handle error analysis accordingly
    # TODO: compute error u1 - u2 - Poiseuille
    # XXX: parabolic inlet profile in narrowed tube needs to be calculated for
    # full tube!

if __name__ == '__main__':
    # DBCvsNitsche_poiseuille()
    # DBCvsNitsche_bench()
    narrow_vs_full_tube()
