''' NS Cavity flow debugging script
    Solve simple cavity flow test case with straight-forward distractionless
    implementations of different 'nonlinear' solvers.

    Parameters:
        U: lid velocity
        N: grid size

        meth: nonlinear solver type
            1: Picard in terms of x_k+1
            2: Picard in terms of update dx_k
            3: Newton in terms of update, started with #2
            4: Picard, Aitken accelerated
            5: Newton, Aitken accelerated, started with #4
        maxit: maximum iterations
        start_maxit: numbers of Picard iteration
        atol: absolute tolerance of (true) residual
        rtol: relative tolerance of increment

        fix_pressure: method of fixing pressure
            0: ignore
            1: substract mean value/specific DOF value
            2: set pointwise 0-DirichletBC in corner specified by geom.Corner()

    output:
        save residual plots in tikz format (TODO)
'''
from dolfin import *
import ufl
# import petsc4py.PETSc as PETSc
from functions.geom import *
import numpy as np
import mshr

parameters["form_compiler"]["precision"] = 100
# stokes error approximately halved w.r.t. value 15
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["cpp_optimize_flags"] = \
    "-O3 -ffast-math -march=native"
# very slight effect on precision of stokes (neither pos nor neg)


def plot_residuals(cases, Res):
    import matplotlib.pyplot as plt
    from matplotlib2tikz import save as tikz_save
    assert len(Res) == len(cases)
    t = 0
    ylabs = [r'$\norm{\delta\vec u_k}_{L_2}/norm{\vec u_k}_{L_2}$',
             r'$\norm{R_k}_{\infty}$',
             r'$\norm{R_k}_{L_2}$',
             r'$\opnorm{R_k}_{L_2}$']
    legs = [r'P',
            r'N',
            r'P+A',
            r'N+A']
    normtype = ['dul2', 'Rinf', 'Rl2', 'Re']
    assert len(ylabs) == len(cases[0][0])
    fignums = plt.get_fignums()
    fig = max(fignums) + 1 if fignums else 0
    # plt.close('all')
    # plt.ion()
    for i, Re in enumerate(Res):
        # caseRe = cases[i][1:]  # ommit first case (P+A u_k+1)
        caseRe = cases[i]  # startet at met=2
        for t in range(len(normtype)):
            for caseMet in caseRe:
                plt.figure(fig)
                plt.semilogy(abs(np.array(caseMet[t])))    # t: normtype index
            fig += 1
            plt.title(r'$Re = {0}$'.format(Re, normtype[t]))
            plt.ylabel(ylabs[t])
            plt.xlabel('iterations')
            plt.legend(legs)
            plt.grid()
            tikz_save(
                'results/dbg_bench/cavity_{0}_Re{1}.tikz'.format(normtype[t],
                                                                 Re),
                figureheight='\\figureheight',
                figurewidth='\\figurewidth')
    # plt.show()


def URe(Re):
    return 0.035*Re/1.0


# TODO: return INF, L2, ENERGY NORM of RESIDUAL (and ||du||), plot
# @profile
def run(meth, Re, start_maxit=5):
    # U = 0.35
    U = URe(Re)
    N = 20
    # use_Re_continuation = False      # use meth = 5 !
    # Ustart = 180.    # start inflow velocity
    # Unum = 20       # continuation steps
    maxit = 100
    fix_pressure = 2    # 0: ignore, 1: correct -p0, 2: point dBC
    # meth = 2
    elem = 1           # 0: TH, 1: Mini
    use_temam = True        # seems to really improve convergence (slightly)

    # atol (based on resid), rtol (based on du)
    rtol = 1e-16
    atol = 1e-13

    stemam = Constant(use_temam)
    mu = Constant(0.035)
    rho = Constant(1.0)

    geo = mshr.Rectangle(Point(0.0, 0.0), Point(1.0, 1.0))
    mesh = mshr.generate_mesh(geo, N)
    ndim = mesh.topology().dim()

    lidflow = Expression(("uin*(1 - pow(x[0] - 0.5, 4)/R4)", "0.0"), uin=U,
                         R4=0.5**4)
    zero = Constant((0,)*ndim)

    if elem == 0:
        V = VectorFunctionSpace(mesh, 'CG', 2)
    elif elem == 1:
        P1 = FunctionSpace(mesh, 'CG', 1)
        B = FunctionSpace(mesh, 'Bubble', ndim + 1)
        V = MixedFunctionSpace(ndim*[P1 + B])

        lidflow = project(lidflow, V, solver_type='mumps')
        zero = project(zero, V, solver_type='mumps')
    Q = FunctionSpace(mesh, 'CG', 1)
    W = V*Q

    bcs = [
        DirichletBC(W.sub(0), lidflow, Top(1.)),
        DirichletBC(W.sub(0), zero, Bottom(0.)),
        DirichletBC(W.sub(0), zero, Left(0.)),
        DirichletBC(W.sub(0), zero, Right(1.)),
    ]
    if fix_pressure == 2:
        bcs.append(DirichletBC(W.sub(1), 0.0, Corner((0., 0.)),
                               method='pointwise'))

    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)

    u0 = Function(V)
    u0.vector().zero()

    a = inner(mu*grad(u), grad(v))*dx + rho*dot(grad(u)*u0, v)*dx - \
        p*div(v)*dx + q*div(u)*dx
    a += stemam*0.5*rho*div(u0)*dot(u, v)*dx

    nc = (rho*dot(grad(u0)*u, v)*dx + stemam*0.5*rho*div(u)*dot(u0, v)*dx)

    L = dot(zero, v)*dx

    A = assemble(a)
    b = assemble(L)
    [bc.apply(A, b) for bc in bcs]

    w = Function(W)
    solve(A, w.vector(), b, 'mumps')

    # adapt pressure constant  (compare against pressure boundary condition)
    (u1, p1) = w.split(deepcopy=True)
    if fix_pressure == 1:
        p0 = p1.vector().array().mean()  # [0]
        p1.vector()[:] -= p0
        assign(w.sub(1), p1)

    # residual
    R = b - A*w.vector()
    print('STOKES residual: {0}'.format(norm(R, 'linf')))

    # 1     Fixed point iteration: Picard x_k+1
    # 2     Fixed point iteration: Picard update
    # 3     Newton update
    # 4     Picard + Aitken
    dunorm_lst = []
    resnorm_inf_lst = []
    resnorm_l2_lst = []
    enorm_lst = []
    it = 0
    if meth == 1:
        it = 0
        while it < maxit:
            assign(u0, w.sub(0))
            A1 = assemble(a)
            [bc.apply(A1, b) for bc in bcs]
            solve(A1, w.vector(), b, 'mumps')

            if fix_pressure == 1:
                (_, p1) = w.split(deepcopy=True)
                p0 = p1.vector().array().mean()
                p1.vector()[:] -= p0
                assign(w.sub(1), p1)

            # true real one and only residual: .. grad(u)*u.
            assign(u0, w.sub(0))
            A1 = assemble(a)
            [bc.apply(A1, b) for bc in bcs]
            resid = b - A1*w.vector()
            print('{1} \t residual: {0}'.format(norm(resid, 'l2'), it))

            it += 1

    elif meth == 2 or meth == 3:
        if meth == 3:
            _maxit = maxit
            maxit = start_maxit
        dw = Function(W)
        it = 0
        bcs_dw = [
            DirichletBC(W.sub(0), zero, "on_boundary"),
            # DirichletBC(W.sub(0), zero, Top(1.0)),
            # DirichletBC(W.sub(0), zero, Bottom(-1.0)),
            # DirichletBC(W.sub(0), zero, Left(0.0)),
        ]
        if fix_pressure == 2:
            bcs_dw.append(DirichletBC(W.sub(1), 0.0, Corner((0.0, 0.0)),
                                      method='pointwise'))
        a1 = 1.0
        while it < maxit:
            assign(u0, w.sub(0))
            A1 = assemble(a)
            rhs = b - A1*w.vector()
            [bc.apply(A1, rhs) for bc in bcs_dw]
            solve(A1, dw.vector(), rhs, 'mumps')

            if fix_pressure == 1:
                (_, dp1) = dw.split(deepcopy=True)
                dp0 = dp1.vector().array().mean()   # [0]
                dp1.vector()[:] -= dp0
                assign(dw.sub(1), dp1)

            w.vector().axpy(1.0, dw.vector())

            # true real one and only residual: .. grad(u)*u.
            assign(u0, w.sub(0))
            A1 = assemble(a)
            [bc.apply(A1, rhs) for bc in bcs]
            resid = b - A1*w.vector()
            resnorm_inf = norm(resid, 'linf')
            resnorm_inf_lst.append(resnorm_inf)
            resnorm_l2 = norm(resid, 'l2')
            resnorm_l2_lst.append(resnorm_l2)
            (du1, _) = dw.split(deepcopy=True)
            dunorm = norm(du1.vector(), 'l2')/norm(u0.vector(), 'l2')
            enorm = np.dot(resid.array(), w.vector().array())
            enorm_lst.append(enorm)
            dunorm_lst.append(dunorm)
            print('{1}\tP\t{2:.2g}\t resid: {0:.4e}\t du: {3:.4e}\t '
                  'En: {4:.4e}'.format(resnorm_inf, it, a1, dunorm, enorm))

            if resnorm_l2 < atol or dunorm < rtol:
                print('converged to tol after {0} iterations'.format(it))
                break
            it += 1
        else:
            print('maxit reached')

    if meth == 3:
        maxit = _maxit
        dw = Function(W)
        bcs = [
            DirichletBC(W.sub(0), zero, "on_boundary"),
            # DirichletBC(W.sub(0), zero, Top(1.0)),
            # DirichletBC(W.sub(0), zero, Bottom(-1.0)),
            # DirichletBC(W.sub(0), zero, Left(0.0)),
        ]
        if fix_pressure == 2:
            bcs.append(DirichletBC(W.sub(1), 0.0, Corner((0.0, 0.0)),
                                   method='pointwise'))

        while it < maxit:
            assign(u0, w.sub(0))
            A1 = assemble(a)
            Nc = assemble(nc)
            rhs = b - A1*w.vector()
            K = A1 + Nc
            [bc.apply(K, rhs) for bc in bcs]
            solve(K, dw.vector(), rhs, 'mumps')

            if fix_pressure == 1:
                (_, dp1) = dw.split(deepcopy=True)
                dp0 = dp1.vector().array().mean()   # [0]
                dp1.vector()[:] -= dp0
                assign(dw.sub(1), dp1)

            w.vector().axpy(1.0, dw.vector())

            # true real one and only residual: .. grad(u)*u.
            assign(u0, w.sub(0))
            A1 = assemble(a)
            rhs = b - A1*w.vector()
            [bc.apply(A1, rhs) for bc in bcs]
            resid = b - A1*w.vector()
            resnorm_inf = norm(resid, 'linf')
            resnorm_inf_lst.append(resnorm_inf)
            resnorm_l2 = norm(resid, 'l2')
            resnorm_l2_lst.append(resnorm_l2)
            (du1, _) = dw.split(deepcopy=True)
            dunorm = norm(du1.vector(), 'l2')/norm(u0.vector(), 'l2')
            enorm = np.dot(resid.array(), w.vector().array())
            enorm_lst.append(enorm)
            dunorm_lst.append(dunorm)
            print('{1}\t\t{2:.2g}\t resid: {0:.4e}\t du: {3:.4e}\t '
                  'En: {4:.4e}'.format(resnorm_inf, it, a1, dunorm, enorm))
            if resnorm_l2 < atol or dunorm < rtol:
                print('converged to tol after {0} iterations'.format(it))
                break

            it += 1
        else:
            print('maxit reached')

    if meth in (4, 5, 6, 7, 8):
        ''' Picard + Aitken accelerator '''
        if meth in (5, 6, 7, 8):
            _maxit = maxit
            maxit = start_maxit
        a1 = 1.0
        dw = Function(W)
        bcs_dw = [
            DirichletBC(W.sub(0), zero, "on_boundary"),
            # DirichletBC(W.sub(0), zero, Top(1.0)),
            # DirichletBC(W.sub(0), zero, Bottom(-1.0)),
            # DirichletBC(W.sub(0), zero, Left(0.0)),
        ]
        if fix_pressure == 2:
            bcs_dw.append(DirichletBC(W.sub(1), 0.0, Corner((0.0, 0.0)),
                                      method='pointwise'))

        (u, p) = split(w)
        while it < maxit:
            assign(u0, w.sub(0))
            A1 = assemble(a)
            rhs = b - A1*w.vector()
            [bc.apply(A1, rhs) for bc in bcs_dw]
            solve(A1, dw.vector(), rhs, 'mumps')

            if fix_pressure == 1:
                (_, dp1) = dw.split(deepcopy=True)
                dp0 = dp1.vector().array().mean()   # [0]
                dp1.vector()[:] -= dp0
                assign(dw.sub(1), dp1)

            (du, dp) = dw.split(deepcopy=True)
            du1 = du.vector().array()
            if it > 0:
                a1 = -a1*np.dot(du0, du1 - du0)/np.linalg.norm(du1 - du0,
                                                               ord=2)**2
            du0 = du1

            w.vector().axpy(a1, dw.vector())

            # true real one and only residual: .. grad(u)*u.
            assign(u0, w.sub(0))
            A0 = assemble(a)
            b = assemble(L)
            # F = inner(mu*grad(u), grad(v))*dx + rho*dot(grad(u)*u, v)*dx - \
            #     p*div(v)*dx + q*div(u)*dx
            # F += stemam*0.5*rho*div(u)*dot(u, v)*dx
            enorm = np.dot((A0*w.vector() - b).array(), w.vector().array())
            enorm_lst.append(enorm)
            [bc.apply(A0, b) for bc in bcs]
            # [bc.apply(F) for bc in bcs]
            resid = A0*w.vector() - b
            # resid = assemble(F)
            # enorm = np.dot(resid.array(), w.vector().array())
            # enorm = assemble(F)
            resnorm_inf = norm(resid, 'linf')
            resnorm_inf_lst.append(resnorm_inf)
            resnorm_l2 = norm(resid, 'l2')
            resnorm_l2_lst.append(resnorm_l2)
            dunorm = np.linalg.norm(du1)/norm(u0.vector(), 'l2')
            dunorm_lst.append(dunorm)
            print('{1}\tP\t{2:.2g}\t resid: {0:.4e}\t du: {3:.4e}\t '
                  'En: {4:.4f}'.format(resnorm_l2, it, a1, dunorm, enorm))
            if resnorm_l2 < atol or dunorm < rtol:
                print('converged to tol after {0} iterations'.format(it))
                break

            it += 1
        else:
            print('maxit reached')

    if meth == 5:
        a1 = 1.0
        maxit = _maxit
        dw = Function(W)
        bcs = [
            DirichletBC(W.sub(0), zero, "on_boundary"),
            # DirichletBC(W.sub(0), zero, Top(1.0)),
            # DirichletBC(W.sub(0), zero, Bottom(-1.0)),
            # DirichletBC(W.sub(0), zero, Left(0.0)),
        ]
        if fix_pressure == 2:
            bcs.append(DirichletBC(W.sub(1), 0.0, Corner((0.0, 0.0)),
                                   method='pointwise'))

        while it < maxit:
            assign(u0, w.sub(0))
            A1 = assemble(a)
            Nc = assemble(nc)
            rhs = b - A1*w.vector()
            K = A1 + Nc
            [bc.apply(K, rhs) for bc in bcs]
            solve(K, dw.vector(), rhs, 'mumps')

            if fix_pressure == 1:
                (_, dp1) = dw.split(deepcopy=True)
                dp0 = dp1.vector().array().mean()   # [0]
                dp1.vector()[:] -= dp0
                assign(dw.sub(1), dp1)

            (du, dp) = dw.split(deepcopy=True)
            du1 = du.vector().array()
            if it > 0:
                a1 = -a1*np.dot(du0, du1 - du0)/np.linalg.norm(du1 - du0,
                                                               ord=2)**2
            du0 = du1

            w.vector().axpy(a1, dw.vector())

            # true real one and only residual: .. grad(u)*u.
            assign(u0, w.sub(0))
            A1 = assemble(a)
            rhs = b - A1*w.vector()
            [bc.apply(A1, rhs) for bc in bcs]
            resid = b - A1*w.vector()
            resnorm_inf = norm(resid, 'linf')
            resnorm_inf_lst.append(resnorm_inf)
            resnorm_l2 = norm(resid, 'l2')
            resnorm_l2_lst.append(resnorm_l2)
            dunorm = np.linalg.norm(du1)/norm(u0.vector(), 'l2')
            dunorm_lst.append(dunorm)
            enorm = np.dot(resid.array(), w.vector().array())
            enorm_lst.append(enorm)
            print('{1}\t\t{2:.2g}\t resid: {0:.4e}\t du: {3:.4e}\t '
                  'En: {4:.4e}'.format(resnorm_l2, it, a1, dunorm, enorm))

            if resnorm_l2 < atol or dunorm < rtol:
                print('converged to tol after {0} iterations'.format(it))
                break

            it += 1
        else:
            print('maxit reached')

    if meth == 6:
        maxit = _maxit
        w = fenics_newton(w, bcs, maxit, atol)

    if meth == 7:
        maxit = _maxit
        w = manual_newton(w, bcs, maxit, atol)

    if meth == 8:
        maxit = _maxit
        w = snes_newton(w, bcs, maxit, atol)

    print('Re = {0}'.format(1.*U/0.035))

    # (ue, pe) = w.split(deepcopy=True)

    # return [dunorm_lst, resnorm_inf_lst, resnorm_l2_lst, enorm_lst]
    return w


def initial_condition(Re):
    # U = 0.35
    U = URe(Re)
    N = 32
    # use_Re_continuation = False      # use meth = 5 !
    # Ustart = 180.    # start inflow velocity
    # Unum = 20       # continuation steps
    fix_pressure = 2    # 0: ignore, 1: correct -p0, 2: point dBC
    # meth = 2
    elem = 1           # 0: TH, 1: Mini
    use_temam = True        # seems to really improve convergence (slightly)

    stemam = Constant(use_temam)
    mu = Constant(0.035)
    rho = Constant(1.0)

    geo = mshr.Rectangle(Point(-1., -1.), Point(1., 1.))
    mesh = mshr.generate_mesh(geo, N)
    ndim = mesh.topology().dim()

    lidflow = Expression(("uin*(1 - pow(x[0], 4))", "0.0"), uin=U)
    zero = Constant((0,)*ndim)

    if elem == 0:
        V = VectorFunctionSpace(mesh, 'CG', 2)
    elif elem == 1:
        P1 = FunctionSpace(mesh, 'CG', 1)
        B = FunctionSpace(mesh, 'Bubble', ndim + 1)
        V = MixedFunctionSpace(ndim*[P1 + B])

        lidflow = project(lidflow, V, solver_type='mumps')
        zero = project(zero, V, solver_type='mumps')
    Q = FunctionSpace(mesh, 'CG', 1)
    W = V*Q

    bcs = [
        DirichletBC(W.sub(0), lidflow, Top(1.0)),
        DirichletBC(W.sub(0), zero, Bottom(-1.0)),
        DirichletBC(W.sub(0), zero, Left(-1.0)),
        DirichletBC(W.sub(0), zero, Right(1.0)),
    ]
    if fix_pressure == 2:
        bcs.append(DirichletBC(W.sub(1), 0.0, Corner((0.0, 0.0)),
                               method='pointwise'))

    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)

    u0 = Function(V)
    u0.vector().zero()

    a = inner(mu*grad(u), grad(v))*dx + rho*dot(grad(u)*u0, v)*dx - \
        p*div(v)*dx + q*div(u)*dx
    a += stemam*0.5*rho*div(u0)*dot(u, v)*dx
    L = dot(zero, v)*dx

    A = assemble(a)
    b = assemble(L)
    [bc.apply(A, b) for bc in bcs]

    w = Function(W)
    solve(A, w.vector(), b, 'mumps')

    return w


class GeneralProblem(NonlinearProblem):
    def __init__(self, F, w, bcs):
        NonlinearProblem.__init__(self)
        self.fform = F
        self.w = w
        self.bcs = bcs
        self.jacobian = derivative(F, w)

    def F(self, b, x):
        assemble(self.fform, tensor=b)
        [bc.apply(b, x) for bc in self.bcs]

    def J(self, A, x):
        assemble(self.jacobian, tensor=A)
        [bc.apply(A) for bc in self.bcs]


def snes_newton(w_init, bcs, maxit, atol):
    use_temam = True        # seems to really improve convergence (slightly)
    Vr = w_init.function_space().sub(0).sub(0)
    if (type(Vr.ufl_element()) ==
            ufl.finiteelement.enrichedelement.EnrichedElement):
        elem = 1
    else:
        elem = 0
    # helps greatly with high Re for Picard, Newton
    # improved too for moderate Re
    # high Re: Aitken + BFS necessary (1000+)
    # WITH backflow stabilization: Newton+Aitken converges, Newton solo stuck
    # at ~ 1e-4 instead of 1e-13
    # NOTE: Newton+Aitken+BF converges to O(R)=1e-13 also at high Re, almost
    # independet of Reynolds number
    # NOTE: Re=5500 (U=180, meth=5, start=15, Temam+BFS, Mini) --> 1e-12
    #       Re=8000 -> converges to 1e-12 when started with 30 (!!) Picard its
    #               ==> gradually approach Reynolds number!

    # NOTE: P+A (duk) always reaches lower tols than P+A (u_k+1)!

    # atol (based on resid), rtol (based on du)

    W = w_init.function_space()
    ndim = W.mesh().topology().dim()

    mu = Constant(0.035)
    rho = Constant(1.0)
    stemam = Constant(use_temam)
    # sbf = Constant(use_backflowstab)

    # n = FacetNormal(W.mesh())

    zero = Constant((0,)*ndim)

    dw = TrialFunction(W)
    z = TestFunction(W)

    # (du, dp) = split(dw)
    (v, q) = split(z)

    w = Function(W)
    (u, p) = split(w)

    # w_init = initial_condition(Re)
    # w0 = Function(W)
    if elem == 0:
        w.interpolate(w_init)
    elif elem == 1:
        w.assign(project(w_init, w.function_space(), solver_type='mumps'))
        zero = project(zero, W.sub(0).collapse(), solver_type='mumps')

    a = inner(mu*grad(u), grad(v))*dx + rho*dot(grad(u)*u, v)*dx - \
        p*div(v)*dx + q*div(u)*dx
    a += stemam*0.5*rho*div(u)*dot(u, v)*dx
    # a += -sbf*rho*0.5*abs_n(dot(u, n))*dot(u, v)*ds
    L = dot(zero, v)*dx

    F = a - L
    # F = action(a - L, w0)
    J = derivative(F, w, dw)

    dunorm_lst = []
    resnorm_inf_lst = []

    PETScOptions().set('snes_monitor')
    PETScOptions().set('snes_newtontr')
    PETScOptions().set('snes_converged_reason')
    PETScOptions().set('snes_atol', 1e-13)
    PETScOptions().set('snes_rtol', 1e-16)
    PETScOptions().set('snes_stol', 0.0)
    PETScOptions().set('snes_max_it', 10)
    PETScOptions().set('ksp_type', 'preonly')
    PETScOptions().set('pc_type', 'lu')
    PETScOptions().set('pc_factor_mat_solver_package', 'mumps')
    problem = GeneralProblem(F, w, bcs=bcs)
    solver = PETScSNESSolver()
    solver.init(problem, w.vector())
    # solver.parameters["linear_solver"] = "mumps"
    # solver.parameters["report"] = False
    # solver.parameters["lu_solver"]["report"] = False
    # solver.parameters["krylov_solver"]["report"] = False
    snes = solver.snes()
    snes.setFromOptions()
    print(snes.getTolerances())
    snes.setConvergenceHistory()
    # snes.logConvergenceHistory(1)
    # snes.setTolerances(rtol=1e-13, atol=1e-13, stol=0.0, max_it=30)
    snes.solve(None, as_backend_type(w.vector()).vec())
    resid = snes.getConvergenceHistory()[0]

    # resid = assemble(F)
    # [bc.apply(resid) for bc in bcs_dw]
    # print('resid: {0}'.format(norm(resid, 'l2')))
    # print('time elapsed: {0}s'.format(toc()))
    Warning("Make exact Newton and other method's output/norms identical")
    return w


def fenics_newton(w_init, bcs, maxit, atol):
    use_temam = True        # seems to really improve convergence (slightly)
    Vr = w_init.function_space().sub(0).sub(0)
    if (type(Vr.ufl_element()) ==
            ufl.finiteelement.enrichedelement.EnrichedElement):
        elem = 1
    else:
        elem = 0
    # helps greatly with high Re for Picard, Newton
    # improved too for moderate Re
    # high Re: Aitken + BFS necessary (1000+)
    # WITH backflow stabilization: Newton+Aitken converges, Newton solo stuck
    # at ~ 1e-4 instead of 1e-13
    # NOTE: Newton+Aitken+BF converges to O(R)=1e-13 also at high Re, almost
    # independet of Reynolds number
    # NOTE: Re=5500 (U=180, meth=5, start=15, Temam+BFS, Mini) --> 1e-12
    #       Re=8000 -> converges to 1e-12 when started with 30 (!!) Picard its
    #               ==> gradually approach Reynolds number!

    # NOTE: P+A (duk) always reaches lower tols than P+A (u_k+1)!

    # atol (based on resid), rtol (based on du)

    W = w_init.function_space()
    ndim = W.mesh().topology().dim()

    mu = Constant(0.035)
    rho = Constant(1.0)
    stemam = Constant(use_temam)
    # sbf = Constant(use_backflowstab)

    # n = FacetNormal(W.mesh())

    zero = Constant((0,)*ndim)

    dw = TrialFunction(W)
    z = TestFunction(W)

    # (du, dp) = split(dw)
    (v, q) = split(z)

    w = Function(W)
    (u, p) = split(w)

    # w_init = initial_condition(Re)
    # w0 = Function(W)
    if elem == 0:
        w.interpolate(w_init)
    elif elem == 1:
        w.assign(project(w_init, w.function_space(), solver_type='mumps'))
        zero = project(zero, W.sub(0).collapse(), solver_type='mumps')

    a = inner(mu*grad(u), grad(v))*dx + rho*dot(grad(u)*u, v)*dx - \
        p*div(v)*dx + q*div(u)*dx
    a += stemam*0.5*rho*div(u)*dot(u, v)*dx
    # a += -sbf*rho*0.5*abs_n(dot(u, n))*dot(u, v)*ds
    L = dot(zero, v)*dx

    F = a - L
    # F = action(a - L, w0)
    J = derivative(F, w, dw)

    dunorm_lst = []
    resnorm_inf_lst = []

    # problem = NSE(J, F)
    # solver = NewtonSolver()
    problem = NonlinearVariationalProblem(F, w, bcs, J=J)
    solver = NonlinearVariationalSolver(problem)
    solver.parameters["newton_solver"]["linear_solver"] = "lu"
    solver.parameters["newton_solver"]["convergence_criterion"] = "residual"  # "incremental"
    solver.parameters["newton_solver"]["relative_tolerance"] = 1e-20
    solver.parameters["newton_solver"]["absolute_tolerance"] = atol
    solver.parameters["newton_solver"]["maximum_iterations"] = maxit
    solver.parameters["newton_solver"]["error_on_nonconvergence"] = False
    solver.solve()

    # resid = assemble(F)
    # [bc.apply(resid) for bc in bcs_dw]
    # print('resid: {0}'.format(norm(resid, 'l2')))
    # print('time elapsed: {0}s'.format(toc()))
    Warning("Make exact Newton and other method's output/norms identical")
    return w


def manual_newton(w_init, bcs, maxit, atol):
    use_temam = True        # seems to really improve convergence (slightly)
    Vr = w_init.function_space().sub(0).sub(0)
    if (type(Vr.ufl_element()) ==
            ufl.finiteelement.enrichedelement.EnrichedElement):
        elem = 1
    else:
        elem = 0
    # helps greatly with high Re for Picard, Newton
    # improved too for moderate Re
    # high Re: Aitken + BFS necessary (1000+)
    # WITH backflow stabilization: Newton+Aitken converges, Newton solo stuck
    # at ~ 1e-4 instead of 1e-13
    # NOTE: Newton+Aitken+BF converges to O(R)=1e-13 also at high Re, almost
    # independet of Reynolds number
    # NOTE: Re=5500 (U=180, meth=5, start=15, Temam+BFS, Mini) --> 1e-12
    #       Re=8000 -> converges to 1e-12 when started with 30 (!!) Picard its
    #               ==> gradually approach Reynolds number!

    # NOTE: P+A (duk) always reaches lower tols than P+A (u_k+1)!

    # atol (based on resid), rtol (based on du)

    W = w_init.function_space()
    ndim = W.mesh().topology().dim()

    mu = Constant(0.035)
    rho = Constant(1.0)
    stemam = Constant(use_temam)
    # sbf = Constant(use_backflowstab)

    # n = FacetNormal(W.mesh())

    zero = Constant((0,)*ndim)

    dw = TrialFunction(W)
    z = TestFunction(W)

    # (du, dp) = split(dw)
    (v, q) = split(z)

    w = Function(W)
    (u, p) = split(w)

    # w_init = initial_condition(Re)
    # w0 = Function(W)
    if elem == 0:
        w.interpolate(w_init)
    elif elem == 1:
        w.assign(project(w_init, w.function_space(), solver_type='mumps'))
        zero = project(zero, W.sub(0).collapse(), solver_type='mumps')

    a = inner(mu*grad(u), grad(v))*dx + rho*dot(grad(u)*u, v)*dx - \
        p*div(v)*dx + q*div(u)*dx
    a += stemam*0.5*rho*div(u)*dot(u, v)*dx
    # a += -sbf*rho*0.5*abs_n(dot(u, n))*dot(u, v)*ds
    L = dot(zero, v)*dx

    F = a - L
    # F = action(a - L, w0)
    J = derivative(F, w, dw)

    dunorm_lst = []
    resnorm_inf_lst = []
    resnorm_l2_lst = []
    enorm_lst = []

    bcs_dw = [
        DirichletBC(W.sub(0), zero, "on_boundary"),
        DirichletBC(W.sub(1), 0.0, Corner((0.0, 0.0)), method='pointwise')
    ]

    aitken = False
    a1 = 1.
    dwk = Function(W)
    dwk.vector()[:] += 1.

    # check residuals
    Ev = inner(mu*grad(u), grad(u))*dx
    Rq = q*div(u)*dx
    # cool that this works! u, p = split(w) --> no need for deepcopies etc
    for it in range(maxit):
        Aj = assemble(J)
        bf = assemble(F)
        enorm = np.inner(bf.array(), w.vector().array())
        enorm_lst.append(enorm)

        # [bc.apply(b) for bc in bcs]  # FIXME ???? no. b considered rhs vec
        # maybe impose 0..
        [bc.apply(Aj, bf, w.vector()) for bc in bcs]   # preferable if not 100%
        #        that w_init matches BCs perfectly!
        # [bc.apply(Aj, bf) for bc in bcs_dw]
        # Aj, bf = assemble_system(J, F, bcs_dw)

        ev = assemble(Ev)
        rq = assemble(Rq).norm('l2')

        dunorm = dwk.vector().norm('l2')/w.vector().norm('l2')
        dunorm_lst.append(dunorm)
        resnorm_inf = bf.norm('linf')
        resnorm_inf_lst.append(resnorm_inf)
        resnorm_l2 = bf.norm('l2')
        resnorm_l2_lst.append(resnorm_l2)
        print('{1}\t relax: {a:.2g}\tresid: {0:.4e}\t du: {2:.4e}\t'
              ' En: {E:.4g} \tEv: {ev:.4f}\tRq: {rq:.4g}'.
              format(resnorm_l2, it, dunorm, a=a1, E=enorm, ev=ev, rq=rq))
        if resnorm_l2 < atol:
            print('converged to tol after {0} iterations'.format(it))
            break

        solve(Aj, dwk.vector(), bf, 'lu')

        # dwk1 = dwk.vector()
        # duk1 = dwk.sub(0)
        (duk1_, _) = dwk.split(deepcopy=True)
        duk1 = duk1_.vector()
        if aitken and it > 0:
            # based on full increment
            # a1 = -a1 * (np.dot(dwk1.array(), dwk1.array() - dwk0.array()) /
            #             norm(dwk1 - dwk0, 'l2')**2)
            # based on velocity only
            a1 = -a1 * (np.dot(duk1.array(), duk1.array() - duk0.array()) /
                        np.linalg.norm(duk1.array() - duk0.array())**2)
        # dwk0 = dwk1.copy()
        duk0 = duk1.copy()

        w.vector().axpy(-a1, dwk.vector())

    else:
        print('maxit reached')

    Warning("Make exact Newton and other method's output/norms identical")
    return w


def loop():
    Re = [100, 1000, 2400, 3600, 4800]
    metds = [2, 3, 4, 5]
    start_maxit = [1, 1, 5, 10, 15]

    # this is stupid.
    cases = []
    for (R, si) in zip(Re, start_maxit):
        case_Re = []
        for met in metds:
            print(R, met, si)
            tmp = run(met, R, start_maxit=si)
            case_Re.append(tmp)
        cases.append(case_Re)

    return cases, Re


if __name__ == '__main__':
    w = run(4, 1000, start_maxit=5)
