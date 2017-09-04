from dolfin import *
# from solvers.ns_iteration import *
from functions.geom import *
import mshr
import numpy as np

set_log_level(ERROR)
parameters["form_compiler"]["precision"] = 100
# set_log_level(1)
# parameters["allow_extrapolation"] = True
# parameters["form_compiler"]["optimize"] = True
# parameters["form_compiler"]["cpp_optimize"] = True
# # parameters["form_compiler"]["representation"] = "quadrature"
# # parameters['num_threads'] = 2
# parameters["form_compiler"]["cpp_optimize_flags"] = \
#     "-O3 -ffast-math -march=native"


def getForms(W, bnds, opt):

    ndim = W.mesh().topology().dim()

    n = FacetNormal(W.mesh())
    h = CellSize(W.mesh())

    k = Constant(opt['mu'])
    rho = Constant(opt['rho'])
    beta = Constant(opt['beta'])
    beta2 = Constant(opt['beta2'])

    elements = opt['elements']
    bctype = opt['bctype']
    wall = opt['wall']
    lidtype = opt['lidtype']
    ulid = opt['ulid']
    # symmetric = opt['symmetric']
    # Rin = opt['Rin']
    # pin = opt['pin']
    sconv = Constant(opt['convection'])
    stem = Constant(opt['temam'])
    snewt = Constant(opt['nonlinear'] == 'newton')
    ssupg = Constant(opt['newton']['supg'])
    spspg = Constant(opt['newton']['pspg'])
    slsic = Constant(opt['newton']['lsic'])

    if bnds:
        # boundary numbering
        # 1   inlet
        bnd_in = 1
        # 2   outlet
        bnd_out = 2
        # 3   wall
        bnd_w = 3
        # 4   slip/symmetry
        bnd_s = 4
    else:
        bnd_t = 2
        bnd_w = 1
        bnds = MeshFunction('size_t', W.mesh(), ndim - 1)
        bnds.set_all(0)
        walls = AllBoundaries()
        top = Top(1.0)
        if lidtype in ['leaky', 'regularized']:
            # this means: x \in [1; 1]
            walls.mark(bnds, bnd_w)
            top.mark(bnds, bnd_t)
        elif lidtype == 'tight':
            # x \in ]1; 1[
            left = Left(-1.0)
            right = Right(1.0)
            bot = Bottom(-1.0)
            top.mark(bnds, bnd_t)
            left.mark(bnds, bnd_w)
            right.mark(bnds, bnd_w)
            bot.mark(bnds, bnd_w)
            # FIXME CONTINUE HERE.
            # top.mark(bnds, bnd_t)
            # top = bnds
        else:
            error('lidtype option "{0}" not recognized.'.format(lidtype))
        # plot(bnds)

    zero = Constant((0,)*ndim)

    bcs = []

    # Define variational problem
    ds = Measure('ds', domain=W.mesh(), subdomain_data=bnds)

    F = rho*zero

    def a(u, v):
        return inner(k*grad(u), grad(v))*dx

    def b(v, q):
        return div(v)*q*dx

    def f(v):
        return dot(F, v)*dx

    def t(u, p):
        return dot(k*grad(u), n) - p*n

    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)

    u0 = Function(W.sub(0).collapse(), name='u0')
    p0 = Function(W.sub(1).collapse(), name='p0')

    # Standard form Stokes equations, Dirichlet RBs
    # choose b(u, q) positive so that a(u,u) > 0
    a1 = a(u, v) - b(v, p) + b(u, q)
    aconv = sconv*rho*dot(grad(u)*u0, v)*dx
    aconv += stem*sconv*0.5*rho*div(u0)*dot(u, v)*dx
    aconv_newton = snewt*sconv*rho*dot(grad(u0)*u, v)*dx
    aconv_newton += snewt*stem*sconv*0.5*rho*div(u)*dot(u0, v)*dx
    L1 = f(v)
    Lconv = dot(zero, v)*dx
    Lconv_newton = dot(zero, v)*dx

    # SUPG explicit
    def resM(u0, p0):
        return F - (rho*grad(u0)*u0 - div(k*grad(u0)) + grad(p0))

    # FIXME: h <- compute longest edge of each element!
    tau_inv = sqrt(4.*dot(u0, u0)/h**2 + (12*k/rho/h**2)**2)
    # tau_inv = 2./h*sqrt(dot(u0, u0))
    tau_s = ssupg/tau_inv
    tau_p = spspg/tau_inv
    tau_l = slsic*sqrt(dot(u0, u0))*h/2.
    a_supg = tau_s*dot(dot(u0, grad(v)), resM(u0, p0))*dx
    a_pspg = tau_p/rho*inner(grad(q), resM(u0, p0))*dx
    a_lsic = tau_l*div(v)*div(u0)*dx

    # outflow: backflow stabilization
    if opt['backflow']:
        def abs_n(x):
            return 0.5*(x - abs(x))

        def abs_p(x):
            return 0.5*(x + abs(x))
        # bfstab = -sconv*rho*0.5*abs_n(dot(u0, n))*dot(u, v)
        # bfstab_newt = snewt*sconv*rho*0.5*abs_n(dot(u, n))*dot(u0, v)
        # FIXME: backflow term always, not only when u.n negative.
        # bfstab = -sconv*rho*0.5*dot(-u0, n)*dot(u, v)
        # bfstab_newt = -sconv*snewt*rho*0.5*dot(-u, n)*dot(u0, v)

        # XXX active backflow stab in true residual
        bfstab = -sconv*rho*0.5*abs_n(dot(u0, n))*dot(u, v)
        # FIXME "explicit" backflow stabilization in jacobian of residual
        bfstab_newt = Constant(0.)*dot(u, v)

        # aconv += bfstab*ds
        # aconv_newton += bfstab_newt*ds

    # inflow
    if wall == 'noslip':
        if lidtype in ['leaky', 'tight']:
            if ndim == 3:
                lid = Constant((ulid, 0.0, 0.0))
            elif ndim == 2:
                lid = Constant((ulid, 0.0))
        elif lidtype == 'regularized':
            if ndim == 3:
                lid = Expression(('ux*(1 - pow(x[0], 4))', '0.0', '0.0'),
                                 ux=ulid)
            elif ndim == 2:
                lid = Expression(('ux*(1 - pow(x[0], 4))', '0.0'), ux=ulid)

        if elements == 'Mini':
            parameters['krylov_solver']['monitor_convergence'] = False
            lid = project(lid, W.sub(0).collapse(), solver_type='lu')
            zero = project(zero, W.sub(0).collapse(), solver_type='lu')

        if bctype == 'dirichlet':
            bc0 = DirichletBC(W.sub(0), zero, walls)
            bc1 = DirichletBC(W.sub(0), lid, top)
            bcs.append(bc0)
            bcs.append(bc1)

        elif bctype == 'nitsche':
            # stress tensor boundary integral
            a1 += - dot(k*grad(u)*n, v)*ds \
                + dot(p*n, v)*ds
            # Nitsche: u = zero, lid (all bounds)
            a1 += beta/h*k*dot(u, v)*ds \
                + beta2/h*dot(u, n)*dot(v, n)*ds
            # RHS u = zero
            L1 += beta/h*k*dot(zero, v)*ds(bnd_w) \
                + beta2/h*dot(zero, n)*dot(v, n)*ds(bnd_w)
            # RHS u = lid
            L1 += beta/h*k*dot(lid, v)*ds(bnd_t) \
                + beta2/h*dot(lid, n)*dot(v, n)*ds(bnd_t)
            # positive 'balance' stability terms
            a1 += + dot(u, k*grad(v)*n)*ds \
                - dot(u, q*n)*ds
            L1 += dot(zero, k*grad(v)*n)*ds(bnd_w) \
                - dot(zero, q*n)*ds(bnd_w)
            L1 += dot(lid, k*grad(v)*n)*ds(bnd_t) \
                - dot(lid, q*n)*ds(bnd_t)

            if opt['backflow']:
                aconv += bfstab*ds(bnd_w)
                aconv_newton += bfstab_newt*ds(bnd_w)

    return (a1, aconv, aconv_newton), (L1, Lconv, Lconv_newton), \
        (a_supg, a_pspg, a_lsic), bcs


def solve_nonlinear(opt, w, a, aconv, L, Lconv, bcs, W, bnds):

    F = a + aconv - L - Lconv
    J = derivative(F, w)

    pde = NonlinearVariationalProblem(F, w, bcs, J)

    solver = NonlinearVariationalSolver(pde)

    prm = solver.parameters
    prm['newton_solver']['relative_tolerance'] = opt['tol']
    prm['newton_solver']['maximum_iterations'] = 100
    prm['newton_solver']['relaxation_parameter'] = 1.0  # 0.2 for h=0.1

    if opt['report'] == 'progress':
        set_log_level(PROGRESS)

    solver.solve()

    u1, p1 = w.split(deepcopy=True)

    return u1, p1


def solve_aitken(opt, a, aconv, L, Lconv, bcs, W):
    ''' Picard iteration with Aitken acceleration
        ref. Kuettler & Wall (2008), "Fixed-point fluid-structure interaction
        solvers with dynamic relaxation"

        u2:     new corrected iterator      u_k+2
        u2s:    prediction                 ~u_k+2
        u1:     old corrected iterator      u_k+1
        w1:     new relaxation parameter    w_k+1
        w0:     old relaxation parameter    w_k
        r2:     residual        r_k+2 = ~u_k+2 - u_k+1
        r1:     old residual    r_k+1 = ~u_k+1 - u_k

        for it > 2:
            - get u2s from F(u1)
            - calculate residual
                    r2 = u2s - u1
            - calculate omega from residuals
                    w1 = -w0*r1/(r2 - r1)
            - calculate corrected/relaxed
                    u2 = u1 + w1*(u2s - u1)
            - set u1 = u2, [u1s = u2s (incl in r2)], r1 = r2, w0 = w1
        repeat.

        for it <= 2:
            - u0: initialize with Stokes (u_k=0)
            - u1s: first NSE correction (~u_k+1)
            - residual: r1 = u1s - u0
            - set w0 (initial)
            - set u1 = u1s or u1 = w0*u1s + (1-w0)*u0
        start loop


    '''

    # FIXME: cf. solve_newton() for more compact algorithm!

    # initialize
    A, bs = assemble_stokes(a, L)
    A1 = A.copy()
    b1 = bs.copy()
    [bc.apply(A1, b1) for bc in bcs]

    a1 = 1.0
    # get initial value == Stokes solution
    u1, p = solveLS(A1, b1, W, 'mumps')
    K, bc = assemble_conv(aconv, Lconv, u1)
    M = A + K
    b = bs + bc
    [bc.apply(M, b) for bc in bcs]
    u2s, p = solveLS(M, b, W, 'mumps')
    r1 = u2s.vector().array() - u1.vector().array()
    # u1.assign(a1*u2s + (1.0 - a1)*u1)
    u1.vector().axpy(a1, u2s.vector() - u1.vector())

    it = 0
    resid = [np.linalg.norm(r1)/norm(u1.vector())]
    while it < opt['maxit'] and resid[-1] > opt['tol']:
        K, bc = assemble_conv(aconv, Lconv, u1)
        M = A + K
        b = bs + bc
        [bc.apply(M, b) for bc in bcs]
        u2s, p = solveLS(M, b, W, 'mumps')
        r2 = u2s.vector().array() - u1.vector().array()
        a1 = -a1*np.dot(r1, r2 - r1)/np.linalg.norm(r2 - r1, ord=2)**2
        # u1.assign(u1 + a1*(u2s - u1))
        u1.vector().axpy(a1, u2s.vector() - u1.vector())

        resid.append(np.linalg.norm(r2, ord=2)/norm(u1.vector()))
        # Rabs = np.linalg.norm(r2, ord=2)
        if opt['report'] == 'progress':
            print('{0}  w1 = {1:.4f} \t resid = {2:g}'.format(it, a1,
                                                              resid[-1]))

        r1 = r2
        # u1.vector().axpy(a1, u2s.vector() - u1.vector())
        # u1.assign(u2)

        it += 1
    # plot(u1, title="u1")
    # plot(u2s, title="u2s")

    if opt['report'] in ['summary', 'progress']:
        if it < opt['maxit']:
            reason = 'converged to tol {tol:.2e}'.format(tol=opt['tol'])
        else:
            plot(u1, "u1")
            plot(u2s, "u schlange")
            reason = 'MAXIT reached, rk = {res}, tol = {tol:.2e}'.\
                format(res=resid[-1], tol=opt['tol'])
            # error(reason)
        print('finished after {0} iterations: '.format(it) + reason)

    # plot(u1)
    # plot(u2s)
    # plot(p)

    return u1, p, resid


def solve_newton(opt, a, aconv, aconv_newton, L, Lconv, Lconv_newton, bcs, W):
    ''' exact Newton method
        - calculate correction or
        - calulate new iterate directly (slightly faster)

        - aitken accelerator
        - start with picard iterations (i.e. 5 with aitken, 10 w/o aitken)
    '''

    # DONE:     1. start with picard
    # DONE:     2. relaxation/acceleration
    # DONE:    3. use REAL residual
    # DOESN'T WORK:    compute res for aitken from w=(u,p), not u only
    # FIXME: --> options dict
    correction = opt['newton']['use_corr']
    # FALSE -> 3% faster but w/o 'true residual'
    # FIXME after a while difference in residual
    # maybe CANCELLATION? -> prefer correction=True
    # FIXME:    correction = False probably has error (lin sys?), resid_true
    #   capped at 1.84433e-12 ...
    # TODO:     let Aitken be based on the error of dw=(du,dp), not du only!
    # TODO:     also, make solveLS return w instead of (u,p) -> slight speed up

    aitken = opt['newton']['use_aitken']
    # start_picard = 5  # number of picard iterations (10 if aitken == False)
    start_picard = opt['newton']['start_picard']
    # TODO: implement energy line search
    linesearch = False

    vtk = True
    if vtk:
        f1 = File('results/cavity/u.pvd')
        f2 = File('results/cavity/p.pvd')

    A, bs = assemble_stokes(a, L)
    A1 = A.copy()
    b1 = bs.copy()
    [bc.apply(A1, b1) for bc in bcs]
    # get initial Stokes solution
    u1, p1 = solveLS(A1, b1, W)

    w = Function(W)
    wd = Function(W)
    u0 = Function(W.sub(0).collapse())

    # make homogeneous Dirichlet BC for update du
    # FIXME: WARNING. This is done on ALL boundaries without checking.
    bc_du = []
    if opt['bctype'] == 'dirichlet':
        zero = Constant((0,)*W.mesh().topology().dim())
        if opt['elements'] == 'Mini':
            zero = project(zero, W.sub(0).collapse())
        bc_du.append(DirichletBC(W.sub(0), zero, "on_boundary"))

    a1 = 1.0
    a0 = 1.0
    it = 0
    resid = [np.inf]
    # resid2 = []
    assign(w.sub(0), u1)
    assign(w.sub(1), p1)
    resid_true = [norm(b1 - A1*w.vector(), 'linf')]
    print(resid_true[-1])
    while resid_true[-1] > opt['tol'] and it < opt['maxit']:
        N, bc = assemble_conv(aconv, Lconv, u1)
        K, bn = assemble_conv(aconv_newton, Lconv_newton, u1)
        if it < start_picard:
            # set Newton convection terms to zero
            K.zero()
            bn.zero()
        DR = A + N + K
        # assign lines = 5% computation time.
        # TODO: make solveLS return w, not (u, p)?
        assign(w.sub(0), u1)
        assign(w.sub(1), p1)

        if not correction:  # compute new iterate u_k+1
            rhs = bs + bc + bn + K*w.vector()
            [bc.apply(DR, rhs) for bc in bcs]
            # calculate true residual for plotting/comparison
            Res = bs + bc + bn - (A + N)*w.vector()
            [bc.apply(Res) for bc in bcs]
            u2, p2 = solveLS(DR, rhs, W)
            r2 = u2.vector().array() - u1.vector().array()
            if it > 0:
                a1 = -a1*np.dot(r1, r2 - r1)/np.linalg.norm(r2 - r1, ord=2)**2
            r1 = r2
            u0.assign(u1)
            if aitken:
                u1.vector().axpy(a1, u2.vector() - u1.vector())
                p1.vector().axpy(a1, p2.vector() - p1.vector())
            else:
                u1.assign(u2)
                p1.assign(p2)
            # resid2.append(norm(u1.vector() - u0.vector())/norm(u1.vector()))
            resid.append(np.linalg.norm(r2, ord=2)/norm(u1.vector(), 'l2'))

            if it > 0:
                resid_true.append(norm(Res, 'l2')/res0)
                # resid_true.append(norm(p1, 'l2'))
            else:
                res0 = norm(Res, 'l2')
            # FIXME which residual to use!?

        else:   # compute update (du, dp)
            R = A + N
            rhs = bs + bc + bn - R*w.vector()
            [bc.apply(DR, rhs) for bc in bc_du]
            du, dp = solveLS(DR, rhs, W)
            r2 = du.vector().array()
            if it > 0:
                a1 = -a1*np.dot(r1, r2 - r1)/np.linalg.norm(r2 - r1, ord=2)**2

            r1 = r2
            if aitken:
                a0 = a1
            if linesearch:
                assign(wd.sub(0), du)
                assign(wd.sub(1), dp)
                # al = -assemble(inner(grad(u1), grad(u1))*dx) / \
                #     assemble(inner(grad(du), grad(du))*dx)
                wdarr = wd.vector().array()
                warr = w.vector().array()
                Aarr = A.array()
                # al = np.sqrt(np.dot(warr.T, np.dot(Aarr, warr) + bs.array()) /
                #              np.dot(wdarr.T, np.dot(Aarr, wdarr) + bs.array()))
                al = -((np.dot(wdarr.T, np.dot(Aarr, warr)) +
                       np.dot(bs.array(), wdarr)) /
                       np.dot(wdarr.T, np.dot(Aarr, wdarr) + bs.array()))
                a0 = al

            u1.vector().axpy(a0, du.vector())
            p1.vector().axpy(a0, dp.vector())
            resid.append(np.linalg.norm(r2, ord=2)/norm(u1.vector(), 'l2'))
            resid_true.append(norm(rhs, 'l2'))

            if vtk:
                f1 << u1, it
                f2 << p1, it

        if opt['report'] == 'progress':
            print('{i} {P}\t w1 = {w:.2g} \t res(ait) = {r:g}'
                  '\t true resid = {r2:g}'.
                  format(i=it, r=resid[-1], P='P' if it < start_picard
                         else ' ', w=a0, r2=resid_true[-1]))
        it += 1

    if opt['report'] in ['summary', 'progress']:
        if it < opt['maxit']:
            reason = 'converged to tol {tol:.2e}'.format(tol=opt['tol'])
        else:
            if correction:
                plot(du, "update")
            reason = 'MAXIT reached, rk = {res}, tol = {tol:.2e}'.\
                format(res=resid[-1], tol=opt['tol'])
            # error(reason)
        print('finished after {0} iterations: '.format(it) + reason)

    return u1, p1, resid_true[1:]


def solve_newtonSUPG(opt, a, aconv, aconv_newton, L, Lconv, Lconv_newton,
                     a_supg, a_pspg, a_lsic, bcs, W):
    ''' exact Newton method
        - calculate correction or
        - calulate new iterate directly (slightly faster)

        - aitken accelerator
        - start with picard iterations (i.e. 5 with aitken, 10 w/o aitken)
    '''

    # DONE:     1. start with picard
    # DONE:     2. relaxation/acceleration
    # DONE:    3. use REAL residual
    # DOESN'T WORK:    compute res for aitken from w=(u,p), not u only
    # FIXME: --> options dict
    correction = opt['newton']['use_corr']
    # FALSE -> 3% faster but w/o 'true residual'
    # FIXME after a while difference in residual
    # maybe CANCELLATION? -> prefer correction=True
    # FIXME:    correction = False probably has error (lin sys?), resid_true
    #   capped at 1.84433e-12 ...
    # TODO:     let Aitken be based on the error of dw=(du,dp), not du only!
    # TODO:     also, make solveLS return w instead of (u,p) -> slight speed up

    aitken = opt['newton']['use_aitken']
    # start_picard = 5  # number of picard iterations (10 if aitken == False)
    start_picard = opt['newton']['start_picard']

    vtk = False
    if vtk:
        f1 = File('results/cavity/u.pvd')
        f2 = File('results/cavity/p.pvd')

    A, bs = assemble_stokes(a, L)
    A1 = A.copy()
    b1 = bs.copy()
    [bc.apply(A1, b1) for bc in bcs]
    # get initial Stokes solution
    u1 = Function(W.sub(0).collapse())
    p1 = Function(W.sub(1).collapse())
    if opt['elements'] == 'P1':
        Wi, _ = getFunctionSpaces(W.mesh(), 'TH')
        (ai, _, _), (Li, _, _), _, bcsi = getForms(Wi, None, opt)
        ui, pi, _ = solve_stokes(opt, ai, Li, bcsi, Wi)
        uii = interpolate(ui, u1.function_space())
        pii = interpolate(pi, p1.function_space())
        u1.assign(uii)
        p1.assign(pii)

    # u1, p1 = solveLS(A1, b1, W)
    # u1.vector()[:] = 1
    # p1.vector()[:] = 1

    w = Function(W)
    u0 = Function(W.sub(0).collapse())

    # make homogeneous Dirichlet BC for update du
    # FIXME: WARNING. This is done on ALL boundaries without checking.
    bc_du = []
    if opt['bctype'] == 'dirichlet':
        zero = Constant((0,)*W.mesh().topology().dim())
        if opt['elements'] == 'Mini':
            zero = project(zero, W.sub(0).collapse())
        bc_du.append(DirichletBC(W.sub(0), zero, "on_boundary"))

    a1 = 1.0
    a0 = 1.0
    it = 0
    resid = [np.inf]
    assign(w.sub(0), u1)
    assign(w.sub(1), p1)
    resid_true = [norm(b1 - A1*w.vector(), 'linf')]
    print(resid_true[-1])
    while resid_true[-1] > opt['tol'] and it < opt['maxit']:
        N, bc = assemble_conv(aconv, Lconv, u1)
        K, bn = assemble_conv(aconv_newton, Lconv_newton, u1)
        Gs = assemble_supg(a_supg, u1, p1)
        Gp = assemble_supg(a_pspg, u1, p1)
        Gl, _ = assemble_conv(a_lsic, Lconv, u1)
        if it < start_picard:
            # set Newton convection terms to zero
            K.zero()
            bn.zero()
        DR = A + N + K
        # assign lines = 5% computation time.
        # TODO: make solveLS return w, not (u, p)?
        assign(w.sub(0), u1)
        assign(w.sub(1), p1)

        if not correction:  # compute new iterate u_k+1
            rhs = bs + bc + bn + K*w.vector()
            [bc.apply(DR, rhs) for bc in bcs]
            # calculate true residual for plotting/comparison
            Res = bs + bc + bn - (A + N)*w.vector()
            [bc.apply(Res) for bc in bcs]
            u2, p2 = solveLS(DR, rhs, W)
            r2 = u2.vector().array() - u1.vector().array()
            if it > 0:
                a1 = -a1*np.dot(r1, r2 - r1)/np.linalg.norm(r2 - r1, ord=2)**2
            r1 = r2
            u0.assign(u1)
            if aitken:
                u1.vector().axpy(a1, u2.vector() - u1.vector())
                p1.vector().axpy(a1, p2.vector() - p1.vector())
            else:
                u1.assign(u2)
                p1.assign(p2)
            # resid2.append(norm(u1.vector() - u0.vector())/norm(u1.vector()))
            resid.append(np.linalg.norm(r2, ord=2)/norm(u1.vector(), 'l2'))

            resid_true.append(norm(Res, 'linf'))

        else:   # compute update (du, dp)
            R = A + N
            rhs = bs + bc + bn - R*w.vector() - (Gs + Gp + Gl)
            [bc.apply(DR, rhs) for bc in bc_du]
            du, dp = solveLS(DR, rhs, W)
            r2 = du.vector().array()
            if it > 0:
                a1 = -a1*np.dot(r1, r2 - r1)/np.linalg.norm(r2 - r1, ord=2)**2

            r1 = r2
            if aitken:
                a0 = a1
            u1.vector().axpy(a1, du.vector())
            p1.vector().axpy(a1, dp.vector())
            if it > 0:
                resid.append(np.linalg.norm(r2, ord=2)/norm(u1.vector(), 'l2'))
            resid_true.append(norm(rhs, 'linf'))

            if vtk:
                f1 << u1, it
                f2 << p1, it

        if opt['report'] == 'progress':
            print('{i} {P}\t w1 = {w:.2g} \t res(ait) = {r:g}'
                  '\t true resid = {r2:g}'.
                  format(i=it, r=resid[-1], P='P' if it < start_picard
                         else ' ', w=a0, r2=resid_true[-1]))
        it += 1

    if opt['report'] in ['summary', 'progress']:
        if it < opt['maxit']:
            reason = 'converged to tol {tol:.2e}'.format(tol=opt['tol'])
        else:
            if correction:
                plot(du, "update")
            reason = 'MAXIT reached, rk = {res}, tol = {tol:.2e}'.\
                format(res=resid[-1], tol=opt['tol'])
            # error(reason)
        print('finished after {0} iterations: '.format(it) + reason)

    return u1, p1, resid_true[1:]


def solve_picard(opt, a, aconv, L, Lconv, bcs, W):
    it = 0
    res_u = np.inf
    res_p = np.inf
    resid = []
    A, bs = assemble_stokes(a, L)
    A1 = A.copy()
    b1 = bs.copy()
    [bc.apply(A1, b1) for bc in bcs]
    # get initial value == Stokes solution
    u0, p0 = solveLS(A1, b1, W, 'mumps', plots=opt['plots'])
    while res_u > opt['tol'] and it <= opt['maxit']:
        K, bc = assemble_conv(aconv, Lconv, u0)
        M = A + K
        b = bs + bc
        [bc.apply(M, b) for bc in bcs]
        u1, p1 = solveLS(M, b, W, 'mumps', plots=opt['plots'])

        res_u = norm(u0.vector() - u1.vector(), 'l2')/norm(u0.vector(), 'l2')
        res_p = norm(p0.vector() - p1.vector(), 'l2')/norm(p0.vector(), 'l2')
        resid.append(res_u)
        if opt['report'] == 'progress':
            print('{0} ||e_u|| = {1}\t ||e_p|| = {2}'.format(it, res_u, res_p))

        u0.assign(u1)
        p0.assign(p1)
        it += 1

    if opt['report'] in ['summary', 'report']:
        if it < opt['maxit']:
            reason = 'converged to tol {tol:e}'.format(tol=opt['tol'])
        else:
            reason = 'MAXIT reached, rk = {res}, tol = {tol:e}'.\
                format(res=res_u, tol=opt['tol'])
        print('finished after {0} iterations: '.format(it) + reason)

    return u1, p1, resid


def solve_stokes(opt, a, L, bcs, W):
    ''' ignore convection term '''
    A, b = assemble_stokes(a, L)
    [bc.apply(A, b) for bc in bcs]
    u1, p1 = solveLS(A, b, W, 'mumps', plots=opt['plots'])

    w = Function(W)
    assign(w.sub(0), u1)
    assign(w.sub(1), p1)

    resid = norm(b - A*w.vector(), 'linf')

    return u1, p1, resid


def getFunctionSpaces(meshfile, elements):
    ''' get all mesh depedent functions '''
    if type(meshfile) == str:
        from functions.inout import read_mesh
        mesh, _, bnds = read_mesh(meshfile)
    elif 'dolfin.cpp.mesh' in str(type(meshfile)):
        mesh = meshfile
        bnds = None
    ndim = mesh.topology().dim()
    if elements == 'TH':
        V = VectorFunctionSpace(mesh, "CG", 2)
        Q = FunctionSpace(mesh, "CG", 1)
        W = V*Q
    elif elements == 'Mini':
        # P1 = VectorFunctionSpace(mesh, "CG", 1)
        # B = VectorFunctionSpace(mesh, "Bubble", ndim + 1)
        P1 = FunctionSpace(mesh, "CG", 1)
        B = FunctionSpace(mesh, 'Bubble', ndim + 1)
        Q = FunctionSpace(mesh, "CG", 1)
        V = MixedFunctionSpace(ndim*[P1 + B])
        W = V*Q
    elif elements == 'P1':
        V = VectorFunctionSpace(mesh, "CG", 1)
        Q = FunctionSpace(mesh, "CG", 1)
        W = V*Q

    # ds = Measure('ds', domain=mesh, subdomain_data=bnds)

    return W, bnds


def assemble_stokes(a, L):
    A = assemble(a)
    b = assemble(L)
    # [bc.apply(A, b) for bc in bcs]

    return A, b


def assemble_conv(aconv, Lconv, u0):
    # extract u0 function from form
    # NAME NEEDS TO BE SET IN FUNCTION(V) DEF!
    acoef = aconv.coefficients()
    aname = [str(ac) for ac in acoef]
    Lcoef = Lconv.coefficients()
    Lname = [str(lc) for lc in Lcoef]
    if 'u0' not in aname:
        error('u0 not found among L/aconv.coefficients()!')
    else:
        idx = aname.index('u0')
        acoef[idx].assign(u0)
        if 'u0' in Lname:
            Lcoef[Lname.index('u0')].assign(u0)

    K = assemble(aconv)
    bc = assemble(Lconv)

    return K, bc


def assemble_supg(a, u0, p0):
    # extract u0 and p0 functions from form
    # NAME NEEDS TO BE SET IN FUNCTION(V) DEF!
    acoef = a.coefficients()
    aname = [str(ac) for ac in acoef]
    if 'u0' not in aname or 'p0' not in aname:
        error('u0 or p0 not found among a.coefficients()!')
    else:
        acoef[aname.index('u0')].assign(u0)
        acoef[aname.index('p0')].assign(p0)

    G = assemble(a)

    return G


def solveLS(A, b, W, solver='mumps', plots=False):
    ''' direct solver '''
    w1 = Function(W)
    solve(A, w1.vector(), b, solver)

    # Split the mixed solution
    (u1, p1) = w1.split(deepcopy=True)

    if plots:
        plot(u, title='velocity')
        plot(p, title='pressure')

    return u1, p1


def __solveLS_numpy(A, b, W):
    A = sp.csr_matrix(A)

    # invert A
    # Ainv = sl.inv(A)

    pass


def __solveLS_PETSc(A, P, b, W):
    w1 = Function(W)
    if solver in ['krylov', 'ksp']:
        if prec == 'schur':
            # PETSc Fieldsplit approach via petsc4py interface
            # Schur complement method
            # create petsc matrices
            # A, b = assemble_system(a1, L1, bcs)

            # create PETSc Krylov solver
            ksp = PETSc.KSP().create()
            # check various CG, GMRES, MINRES implementations
            ksp.setType(PETSc.KSP.Type.MINRES)

            # create PETSc FIELDSPLIT preconditioner
            pc = ksp.getPC()
            pc.setType(PETSc.PC.Type.FIELDSPLIT)
            # create index sets (IS)
            is0 = PETSc.IS().createGeneral(W.sub(0).dofmap().dofs())
            is1 = PETSc.IS().createGeneral(W.sub(1).dofmap().dofs())
            fields = [('0', is0), ('1', is1)]
            pc.setFieldSplitIS(*fields)

            pc.setFieldSplitType(PETSc.PC.CompositeType.SCHUR)
            #                          0: additive (Jacobi)
            #                          1: multiplicative (Gauss-Seidel)
            #                          2: symmetric_multiplicative (symGS)
            #                          3: schur
            #                          see PETSc manual p. 92
            # https://www.mcs.anl.gov/petsc/petsc-3.7/docs/manualpages/PC/PCFIELDSPLIT.html#PCFIELDSPLIT
            # pc.setFieldSplitSchurFactType(PETSc.PC.SchurFactType.FULL)
            # <diag,lower,upper,full>
            # == PETSc.PC.SchurFactType.DIAG
            # https://www.mcs.anl.gov/petsc/petsc-3.7/docs/manualpages/PC/PCFieldSplitSetSchurFactType.html#PCFieldSplitSetSchurFactType
            Sp = Sp.getSubMatrix(is1, is1)
            pc.setFieldSplitSchurPreType(PETSc.PC.SchurPreType.USER, Sp)
            # == PETSc.PC.SchurPreType.USER or A11, UPPER, FULL...

            # subksps = pc.getFieldSplitSubKSP()
            # # 'A' velocity block: precondition with hypre_AMG
            # subksps[0].setType('preonly')  # options: ksp.Type.
            # subksps[0].getPC().setType('hypre')  # options: PETSc.PC.Type.

            # # diag(Q) spectrally equivalent to S; Jacobi precond = diagonal
            # #  scaling, using Q
            # subksps[1].setType('preonly')
            # subksps[1].getPC().setType('jacobi')

            # ## NOTE: schur factorization necessary? What difference if I use
            #       P = assemble(bp1) with corresp. fieldsplits and sub PCs?

            ksp.setOperators(A)
            ksp.setFromOptions()
            ksp.setUp()
            (x, _) = A.getVecs()
            ksp.solve(b, x)

            w1.vector()[:] = x.array

        elif prec == 'jacobi':
            # http://fenicsproject.org/qa/5287/using-the-petsc-pcfieldsplit-in-fenics?show=5320#a5320
            A = as_backend_type(A).mat()
            b = as_backend_type(b).vec()
            P = as_backend_type(P).mat()

            # fieldsplit solve
            ksp = PETSc.KSP().create()
            ksp.setType(PETSc.KSP.Type.MINRES)
            pc = ksp.getPC()
            pc.setType(PETSc.PC.Type.FIELDSPLIT)
            is0 = PETSc.IS().createGeneral(W.sub(0).dofmap().dofs())
            is1 = PETSc.IS().createGeneral(W.sub(1).dofmap().dofs())
            pc.setFieldSplitIS(('0', is0), ('1', is1))
            pc.setFieldSplitType(0)  # 0=additive

            # # hacking ------->
            # pc.setFieldSplitType(PETSc.PC.CompositeType.SCHUR)  # 0=additive
            # pc.setFieldSplitSchurFactType(PETSc.PC.SchurFactType.DIAG)
            # # <diag,lower,upper,full>
            # # == PETSc.PC.SchurFactType.DIAG
            # https://www.mcs.anl.gov/petsc/petsc-3.7/docs/manualpages/PC/PCFieldSplitSetSchurFactType.html#PCFieldSplitSetSchurFactType
            # Sp = P.getSubMatrix(is1, is1)
            # pc.setFieldSplitSchurPreType(3, Sp)
            # # == PETSc.PC.SchurPreType.USER or A11, UPPER, FULL...
            # # <----------
            # # subksps = pc.getFieldSplitSubKSP()
            # CRASHES WITH SEGFAULT, SET
            # #   FIELD SPLIT KSPS VIA --petsc. ARGUMENTS
            # # subksps[0].setType("preonly")
            # # subksps[0].getPC().setType("hypre")
            # # subksps[1].setType("preonly")
            # # subksps[1].getPC().setType("jacobi")
            # ksp.setOperators(A)

            # subksps = pc.getFieldSplitSubKSP()
            # subksps[0].setType("preonly")
            # subksps[0].getPC().setType("hypre")
            # subksps[1].setType("preonly")
            # subksps[1].getPC().setType("jacobi")
            ksp.setOperators(A, P)

            ksp.setFromOptions()
            (x, _) = A.getVecs()
            ksp.solve(b, x)

            w1.vector()[:] = x.array

        elif prec == 'direct':
            # XXX ACCOUNT FOR CONSTANT PRESSURE NULLSPACE!?
            # cf. Elman (2005) book pp. 83

            solv = KrylovSolver(solver, prec)
            solv.set_operators(A, P)
            solv.solve(w1.vector(), b)
        (u1, p1) = w1.split(deepcopy=True)
    elif solver in ['mumps', 'lu', 'supderlu_dist', 'umfpack', 'petsc']:
        # use direct solver
        solve(A, w1.vector(), b, solver)

        # Split the mixed solution
        (u1, p1) = w1.split(deepcopy=True)

    unorm = u1.vector().norm("l2")
    pnorm = p1.vector().norm("l2")

    if MPI.COMM_WORLD.rank == 0:
        print("Norm of velocity coefficient vector: %.6g" % unorm)
        print("Norm of pressure coefficient vector: %.6g" % pnorm)
        print("DOFS: %d" % W.dim())

    if plots:
        plot(u1, title='velocity, dirichlet')
        plot(p1, title='pressure, dirichlet')

    return u1, p1, W.dim()


def prep_mesh(mesh_file):
    if on_cluster():
        # running on NLHPC cluster
        mfile = '/home/dnolte/fenics/nitsche/meshes/' + mesh_file
        mesh = '/dev/shm/' + mesh_file
        shutil.copy(mfile, '/dev/shm/')
    else:
        # running on local machine
        mesh = 'meshes/' + mesh_file

    return mesh


def solve_iteration(opt, W, bnds):
    ''' wrapper for different iterative solvers '''
    (a, aconv, aconv_newton), (L, Lconv, Lconv_newton), \
        (a_supg, a_pspg, a_lsic), bcs = getForms(W, bnds, opt)
    if opt['nonlinear'] == 'stokes':
        u, p, resid = solve_stokes(opt, a, L, bcs, W)
    if opt['nonlinear'] == 'picard':
        u, p, resid = solve_picard(opt, a, aconv, L, Lconv, bcs, W)
    elif opt['nonlinear'] == 'aitken':
        u, p, resid = solve_aitken(opt, a, aconv, L, Lconv, bcs, W)
    elif opt['nonlinear'] == 'newton':
        # u, p, resid = solve_newton(opt, a, aconv, aconv_newton, L, Lconv,
        #                            Lconv_newton, bcs, W)
        u, p, resid = solve_newtonSUPG(opt, a, aconv, aconv_newton, L, Lconv,
                                       Lconv_newton, a_supg, a_pspg, a_lsic,
                                       bcs, W)
    elif opt['nonlinear'] == 'fenics':
        w, a, aconv, L, Lconv, bcs = getForms_NonLin(W, bnds, opt)
        u, p = solve_nonlinear(opt, w, a, aconv, L, Lconv, bcs, W, bnds)
        resid = None

    return u, p, resid


def cavity(opt):
    # mesh = UnitSquareMesh(opt['Nmesh'], opt['Nmesh'])
    geo = mshr.Rectangle(Point(-1, -1), Point(1, 1))
    mesh = mshr.generate_mesh(geo, opt['Nmesh'])
    W, _ = getFunctionSpaces(mesh, opt['elements'])

    u1, p1, resid = solve_iteration(opt, W, None)

    return u1, p1, resid

if __name__ == '__main__':
    opt = {
        'Nmesh': 32,
        'elements': 'P1',
        'backflow': False,      # FIXME not implemented correctly for newton
        'convection': True,
        'temam': True,
        'bctype': 'nitsche',
        'wall': 'noslip',
        'lidtype': 'regularized',        # leaky, tight, regularized
        'ulid': 10.,
        'beta': 1e2,               # needs to be large 1e6 ++ for coarc2ds_d0
        'beta2': 10,
        'nonlinear': 'newton',  # newton, picard, aitken, fenics, stokes
        'newton': {
            'use_corr': True,       # false -> can be faster, but more
            # unreliable due to cancellation(?)
            'use_aitken': False,
            'start_picard': 500,
            'supg': True,        # FIXME: ? no improvement. should be P1/P1.
            'pspg': True,        # FIXME: not working
            'lsic': False        # FIXME: not working
        },
        'report': 'progress',         # progress, summary, 'none'
        'maxit': 100,
        'tol': 1.0e-16,
        'mu': 0.035,     # in P = g/cm/s
        'rho': 1.0,  # in g/cm^3  (1 g/cm^3 = 1000 kg/m^3)
        'plots': False,
        'savenpz': True,
        'npzfile': 'results/nse_estim/nse_gm_ui120.npz',
        'savevtk': False
    }

    u, p, resid = cavity(opt)

    Re = opt['rho']*opt['ulid']*2./opt['mu']
    print('Re = {0}'.format(Re))

    plot(u)
    plot(p)

    print('norm(u):\t {0}'.format(norm(u, 'l2')))
    print('norm(p):\t {0}'.format(norm(p, 'l2')))
