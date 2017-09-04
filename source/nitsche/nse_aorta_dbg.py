''' Simple monolithic & direct NSE solver.
    Distraction free debugging script for the aorta test case.
    Options:
        elem:   0: TH elements, 1: Mini elements
        meth:   1: Picard/Aitken u_k+1
                2: Picard w/o Aitken, du_k
                3: Newton + #2 start
                4: Picard w/ Aitken, du_k
                5: Newton w/ Aitken, #4 start
        use_temam: Bool switch for Temam stabilization term
        use_backflowstab:  Bool switch for BFS. No Newton extra terms, makes
                            Newton's method a "quasi Newton" method
        use_Re_continuation:  Bool switch for Reynolds continuation, using
                    solver #5 (Newton+Aitken) with #1 (Picard+Aitken+u_k+1
                    (=>BCs for U)
        Ustart: lowest velocity for Re continuation
        Unum:   steps in Re continuation
        outflow:  0: inflow profile, 1: zero stress
        fix_pressure:  0: ignore, 1: substract p.mean(), 2: 0-DBC in corner
        maxit:  maximum number of iterations
        start_maxit:  number of Picard iterations for starting Newton
        rtol:  relative tolerance of velocity increment
        atol:  absolute tolerance of residual
'''
from dolfin import *
import ufl
import numpy as np
from functions.geom import Corner
from functions.inout import read_mesh

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
        caseRe = cases[i][1:]  # ommit first case (P+A u_k+1)
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
                'results/dbg_bench/aorta_{0}_Re{1}.tikz'.format(normtype[t],
                                                                Re),
                figureheight='\\figureheight',
                figurewidth='\\figurewidth')
    # plt.show()


def abs_n(x):
    return 0.5*(x - abs(x))
    # return -x


def URe(Re):
    return 0.035*Re


# TODO: return INF, L2, ENERGY NORM of RESIDUAL (and ||du||), plot
def run(meth, Re):
    Warning('in meth 4,5 changed conv crit to _l2 for comparing'
            'with fenics newton')
    # U = URe(Re)
    U = 90
    use_Re_continuation = False      # use meth = 5 !
    Ustart = 180.    # start inflow velocity
    Unum = 20       # continuation steps
    maxit = 40
    start_maxit = 10
    fix_pressure = 0    # 0: ignore, 1: correct -p0, 2: point dBC
    outflow = 1         # 0: outflow = inflow, 1: pressure outlet (0 stress)
    elem = 1           # 0: TH, 1: Mini
    use_temam = True        # seems to really improve convergence (slightly)
    use_backflowstab = True
    rtol = 1e-16
    atol = 1e-14
    # helps greatly with high Re for Picard, Newton
    # improved too for moderate Re
    # high Re: Aitken + BFS necessary (1000+)
    # WITH backflow stabilization: Newton+Aitken converges, Newton w/o Aitken
    # stuck at ~ 1e-4 instead of 1e-13
    # NOTE: Newton+Aitken+BF converges to O(R)=1e-13 also at high Re, almost
    # independet of Reynolds number
    # NOTE: Re=5500 (U=180, meth=5, start=15, Temam+BFS, Mini) --> 1e-12
    #       Re=8000 -> converges to 1e-12 when started with 30 (!!) Picard its
    #               ==> gradually approach Reynolds number!

    # NOTE: P+A (duk) always reaches lower tols than P+A (u_k+1)!

    h = 0.05
    mesh, _, bnds = read_mesh('./meshes/coarc2ds_f0.6_d0_h{h}.h5'.format(h=h))
    ndim = mesh.topology().dim()

    if use_Re_continuation and not meth == 5:
        raise Exception("use_Re_continuation requires 'meth == 5'")

    mu = Constant(0.035)
    rho = Constant(1.0)
    stemam = Constant(use_temam)
    sbf = Constant(use_backflowstab)

    zero = Constant((0,)*ndim)
    # inflow = Expression(("uin*4*(0.5*0.5 - pow(x[1] - 0.5, 2))", "0.0"),
    # uin=U)
    Inflow = Expression(("uin*(1 - pow(x[1], 2))", "0.0"), uin=U)
    c0 = Constant(0.0)
    if elem == 0:
        V = VectorFunctionSpace(mesh, 'CG', 2)
        inflow = Inflow
    elif elem == 1:
        P1 = FunctionSpace(mesh, 'CG', 1)
        B = FunctionSpace(mesh, 'Bubble', ndim + 1)
        V = MixedFunctionSpace(ndim*[P1 + B])

        inflow = project(Inflow, V, solver_type='mumps')
        zero = project(zero, V, solver_type='mumps')
        c0 = project(c0, V.sub(1).collapse(), solver_type='mumps')

    Q = FunctionSpace(mesh, 'CG', 1)
    W = V*Q

    bcs = [
        DirichletBC(W.sub(0), zero, bnds, 3),
        # DirichletBC(W.sub(0), inflow, bnds, 4),
        DirichletBC(W.sub(0).sub(1), c0, bnds, 4),
        DirichletBC(W.sub(0), inflow, bnds, 1),
    ]
    if outflow == 0:
        bcs.append(DirichletBC(W.sub(0), inflow, bnds, 2))
    elif outflow == 1:
        pass

    if fix_pressure == 2:
        corn = Corner(4., 0.)
        bcs.append(DirichletBC(W.sub(1), 0.0, corn, method='pointwise'))

    n = FacetNormal(mesh)

    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)

    u0 = Function(V)
    u0.vector().zero()

    a = (inner(mu*grad(u), grad(v))*dx + rho*dot(grad(u)*u0, v)*dx -
         p*div(v)*dx + q*div(u)*dx)
    a += stemam*0.5*rho*div(u0)*dot(u, v)*dx
    a += -sbf*rho*0.5*abs_n(dot(u0, n))*dot(u, v)*ds
    L = dot(zero, v)*dx

    A = assemble(a)
    b = assemble(L)

    if use_Re_continuation:
        if elem == 1:
            Inflow.uin = Ustart
            inflow = project(Inflow, V)
        else:
            inflow.uin = Ustart

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

    # w.vector()[:] *= 1.2

    ''' notes:
        - #1 und #2 not quite identical
        - P+A (#4) huge improvement over P (#1,2)
        - P+N (#3) slightly better convergence than P+N+A (#5) !
    '''
    # TODO: 4. backflow

    # 1     Fixed point iteration: Picard x_k+1 + Aitken for Re continuation
    # 2     Fixed point iteration: Picard update
    # 3     Newton update
    # 4     Picard + Aitken
    # 5     Newton + Aitken
    dunorm_lst = []
    resnorm_inf_lst = []
    resnorm_l2_lst = []
    enorm_lst = []
    if use_Re_continuation:
        Uarr = np.linspace(Ustart, U, num=Unum, endpoint=True)
    else:
        Uarr = [U]
    for k, uin in enumerate(Uarr):
        if use_Re_continuation:
            print('Re continuation {0}/{1}: \t U = {2}'.format(k, Unum-1, uin))
        it = 0

        if use_Re_continuation:
            if elem == 1:
                Inflow.uin = uin
                inflow = project(Inflow, V)
                bcs[2] = DirichletBC(W.sub(0), inflow, bnds, 1)
            else:
                inflow.uin = uin

        if meth == 1 or (meth == 5 and use_Re_continuation):
            if meth == 5:
                _maxit = maxit
                maxit = start_maxit
                if uin > 260:
                    # with maxit = 25, start_maxit = 15, this allows to solve
                    # u = 280 (Re = 16.000) to 1e-10 (and deeper with maxit++)
                    start_maxit += 5
                    _maxit += 5
            w0 = Function(w.function_space())
            a1 = 1.0
            while it < maxit:
                assign(u0, w.sub(0))
                assign(w0, w)
                A1 = assemble(a)
                [bc.apply(A1, b) for bc in bcs]
                solve(A1, w.vector(), b, 'mumps')

                if fix_pressure == 1:
                    (_, p1) = w.split(deepcopy=True)
                    p0 = p1.vector().array().mean()
                    p1.vector()[:] -= p0
                    assign(w.sub(1), p1)

                # aitken >>
                (u1, _) = w.split(deepcopy=True)
                du1 = u1.vector().array() - u0.vector().array()
                if it > 0:
                    a1 = -a1*np.dot(du0, du1 - du0)/np.linalg.norm(du1 - du0,
                                                                   ord=2)**2
                    # w.vector()[:] = w0.vector()[:] + a1*(w.vector()[:] -
                    #                                      w0.vector()[:])
                    # w0.vector()[:] += a1*(w.vector()[:] - w0.vector()[:])
                    w0.vector().axpy(a1, w.vector() - w0.vector())
                    assign(w, w0)
                du0 = du1
                # << aitken

                # true real one and only residual: .. grad(u)*u.
                assign(u0, w.sub(0))
                A1 = assemble(a)
                [bc.apply(A1, b) for bc in bcs]
                resid = b - A1*w.vector()
                resnorm_inf = norm(resid, 'linf')
                resnorm_inf_lst.append(resnorm_inf)
                resnorm_l2 = norm(resid, 'l2')
                resnorm_l2_lst.append(resnorm_l2)
                dunorm = np.linalg.norm(du1)/norm(u0.vector(), 'l2')
                dunorm_lst.append(dunorm)
                enorm = np.dot(resid.array(), w.vector().array())
                enorm_lst.append(enorm)
                print('{1}\tP\t{2:.2g}\t resid: {0:.4e}\t du: {3:.4e}\t '
                      'En: {4:.4e}'.format(resnorm_inf, it, a1, dunorm, enorm))

                if resnorm_inf < atol and dunorm < rtol:
                    print('converged to tol after {0} iterations'.format(it))
                    break

                it += 1
            else:
                print('maxit reached')

        elif meth == 2 or meth == 3:
            if meth == 3:
                _maxit = maxit
                maxit = start_maxit
            it = 0
            # dw = Function(W)
            # bcs_dw = homogenize(bcs)
            bcs_dw = [
                # DirichletBC(W.sub(0), zero, "on_boundary"),
                DirichletBC(W.sub(0), zero, bnds, 1),
                DirichletBC(W.sub(0), zero, bnds, 3),
                DirichletBC(W.sub(0).sub(1), c0, bnds, 4),
            ]
            if outflow == 0:
                bcs_dw.append(DirichletBC(W.sub(0), zero, bnds, 2))
            elif outflow == 1:
                pass
            if fix_pressure == 2:
                bcs_dw.append(DirichletBC(W.sub(1), 0.0, corn,
                                          method='pointwise'))
            # nc = rho*dot(grad(u0)*u, v)*dx
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

                # true real one and only resid: .. grad(u)*u.
                assign(u0, w.sub(0))
                A1 = assemble(a)
                [bc.apply(A1, rhs) for bc in bcs_dw]
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

                if resnorm_inf < atol and dunorm < rtol:
                    print('converged to tol after {0} iterations'.format(it))
                    break

                it += 1
            else:
                print('maxit reached')

        if meth == 3:
            ''' Newton newton '''
            maxit = _maxit
            dw = Function(W)
            bcs_dw = [
                # DirichletBC(W.sub(0), zero, "on_boundary"),
                DirichletBC(W.sub(0), zero, bnds, 1),
                DirichletBC(W.sub(0), zero, bnds, 3),
                DirichletBC(W.sub(0).sub(1), c0, bnds, 4),
            ]
            if outflow == 0:
                bcs_dw.append(DirichletBC(W.sub(0), zero, bnds, 2))
            elif outflow == 1:
                pass
            if fix_pressure == 2:
                bcs_dw.append(DirichletBC(W.sub(1), 0.0, corn,
                                          method='pointwise'))

            nc = (rho*dot(grad(u0)*u, v)*dx +
                  stemam*0.5*rho*div(u)*dot(u0, v)*dx)  # -
            #      # sbf*abs_n(dot(u, n))*dot(u0, v)*ds)
            while it < maxit:
                assign(u0, w.sub(0))
                A1 = assemble(a)
                Nc = assemble(nc)
                rhs = b - A1*w.vector()
                K = A1 + Nc
                [bc.apply(K, rhs) for bc in bcs_dw]
                solve(K, dw.vector(), rhs, 'mumps')

                if fix_pressure == 1:
                    (_, dp1) = dw.split(deepcopy=True)
                    dp0 = dp1.vector().array().mean()   # [0]
                    dp1.vector()[:] -= dp0
                    assign(dw.sub(1), dp1)

                w.vector().axpy(1.0, dw.vector())

                # true real one and only resid: .. grad(u)*u.
                assign(u0, w.sub(0))
                A1 = assemble(a)
                rhs = b - A1*w.vector()
                [bc.apply(A1, rhs) for bc in bcs_dw]
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
                if resnorm_inf < atol and dunorm < rtol:
                    print('converged to tol after {0} iterations'.format(it))
                    break

                it += 1
            else:
                print('maxit reached')

        if meth in (4, 6, 7) or (meth == 5 and not use_Re_continuation):
            ''' Picard + Aitken accelerator '''
            if meth in (5, 6, 7):
                _maxit = maxit
                maxit = start_maxit
            a1 = 1.0
            dw = Function(W)
            bcs_dw = [
                # DirichletBC(W.sub(0), zero, "on_boundary"),
                DirichletBC(W.sub(0), zero, bnds, 1),
                DirichletBC(W.sub(0), zero, bnds, 3),
                DirichletBC(W.sub(0).sub(1), c0, bnds, 4),
            ]
            if outflow == 0:
                bcs_dw.append(DirichletBC(W.sub(0), zero, bnds, 2))
            elif outflow == 1:
                pass
            if fix_pressure == 2:
                bcs_dw.append(DirichletBC(W.sub(1), 0.0, corn,
                                          method='pointwise'))
            # nc = rho*dot(grad(u0)*u, v)*dx
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

                # true real one and only resid: .. grad(u)*u.
                assign(u0, w.sub(0))
                A1 = assemble(a)
                [bc.apply(A1, rhs) for bc in bcs_dw]
                resid = b - A1*w.vector()
                resnorm_inf = norm(resid, 'linf')
                resnorm_inf_lst.append(resnorm_inf)
                resnorm_l2 = norm(resid, 'l2')
                resnorm_l2_lst.append(resnorm_l2)
                dunorm = np.linalg.norm(du1)/norm(u0.vector(), 'l2')
                dunorm_lst.append(dunorm)
                enorm = np.dot(resid.array(), w.vector().array())
                enorm_lst.append(enorm)
                print('{1}\tP\t{2:.2g}\t resid: {0:.4e}\t du: {3:.4e}\t '
                      'En: {4:.4e}'.format(resnorm_l2, it, a1, dunorm, enorm))
                if resnorm_inf < atol and dunorm < rtol:
                    print('converged to tol after {0} iterations'.format(it))
                    break

                it += 1
            else:
                print('maxit reached')

        if meth == 5:
            tic()
            ''' Newton newton '''
            maxit = _maxit
            a1 = 1.0
            dw = Function(W)
            bcs_dw = [
                # DirichletBC(W.sub(0), zero, "on_boundary"),
                DirichletBC(W.sub(0), zero, bnds, 1),
                DirichletBC(W.sub(0), zero, bnds, 3),
                DirichletBC(W.sub(0).sub(1), c0, bnds, 4),
            ]
            if outflow == 0:
                bcs_dw.append(DirichletBC(W.sub(0), zero, bnds, 2))
            elif outflow == 1:
                pass
            if fix_pressure == 2:
                bcs_dw.append(DirichletBC(W.sub(1), 0.0, corn,
                                          method='pointwise'))

            nc = (rho*dot(grad(u0)*u, v)*dx +
                  stemam*0.5*rho*div(u)*dot(u0, v)*dx)  # -
            #      # sbf*abs_n(dot(u, n))*dot(u0, v)*ds)
            while it < maxit:
                assign(u0, w.sub(0))
                A1 = assemble(a)
                Nc = assemble(nc)
                rhs = b - A1*w.vector()
                K = A1 + Nc
                [bc.apply(K, rhs) for bc in bcs_dw]
                solve(K, dw.vector(), rhs, 'mumps')

                if fix_pressure == 1:
                    (_, dp1) = dw.split(deepcopy=True)
                    dp0 = dp1.vector().array().mean()   # [0]
                    dp1.vector()[:] -= dp0
                    assign(dw.sub(1), dp1)

                (du, dp) = dw.split(deepcopy=True)
                du1 = du.vector().array()
                if it > start_maxit:
                    a1 = -a1*np.dot(du0, du1 - du0)/np.linalg.norm(du1 - du0,
                                                                   ord=2)**2
                du0 = du1
                # a1 = 1.0
                w.vector().axpy(a1, dw.vector())

                # true real one and only resid: .. grad(u)*u.
                # FIXME: this is not really necessary
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
                      'En: {4:g}'.format(resnorm_l2, it, a1, dunorm, enorm))

                if resnorm_l2 < atol or dunorm < rtol:
                    print('converged to tol after {0} iterations'.format(it))
                    break

                it += 1
            else:
                print('maxit reached')
            print('time elapsed: {0}s'.format(toc()))

        if meth == 6:
            maxit = _maxit
            fenics_newton(w, bcs, bnds, maxit, atol)

        if meth == 7:
            maxit = _maxit
            manual_newton(w, bcs, bnds, maxit, atol)

        print('Re = {0}'.format(1.*uin/0.035))

    (ue, pe) = w.split(deepcopy=True)

    return w

    # return [dunorm_lst, resnorm_inf_lst, resnorm_l2_lst, enorm_lst]


def fenics_newton(w_init, bcs, bnds, maxit, atol):
    use_temam = True        # seems to really improve convergence (slightly)
    use_backflowstab = True
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
    sbf = Constant(use_backflowstab)

    n = FacetNormal(W.mesh())

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
        w.assign(project(w_init, w.function_space()))
        zero = project(zero, W.sub(0).collapse(), solver_type='mumps')

    a = inner(mu*grad(u), grad(v))*dx + rho*dot(grad(u)*u, v)*dx - \
        p*div(v)*dx + q*div(u)*dx
    a += stemam*0.5*rho*div(u)*dot(u, v)*dx
    a += -sbf*rho*0.5*abs_n(dot(u, n))*dot(u, v)*ds
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
    return w


def manual_newton(w_init, bcs, bnds, maxit, atol):
    use_temam = True        # seems to really improve convergence (slightly)
    use_backflowstab = True
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
    sbf = Constant(use_backflowstab)

    n = FacetNormal(W.mesh())

    zero = Constant((0,)*ndim)

    dw = TrialFunction(W)
    z = TestFunction(W)

    # (du, dp) = split(dw)
    (v, q) = split(z)

    w = Function(W)
    (u, p) = split(w)

    # w_init = initial_condition(Re)
    # w0 = Function(W)
    c0 = Constant(0.0)
    if elem == 0:
        w.interpolate(w_init)
    elif elem == 1:
        w.assign(project(w_init, w.function_space()))
        zero = project(zero, W.sub(0).collapse(), solver_type='lu')
        c0 = project(c0, W.sub(0).sub(1).collapse(), solver_type='lu')

    a = inner(mu*grad(u), grad(v))*dx + rho*dot(grad(u)*u, v)*dx - \
        p*div(v)*dx + q*div(u)*dx
    a += stemam*0.5*rho*div(u)*dot(u, v)*dx
    a += -sbf*rho*0.5*abs_n(dot(u, n))*dot(u, v)*ds   # v = 0 on all
    #       Dirichlet bounds anyways
    L = dot(zero, v)*dx

    F = a - L
    # F = action(a - L, w0)
    J = derivative(F, w, dw)

    dunorm_lst = []
    resnorm_inf_lst = []
    resnorm_l2_lst = []
    enorm_lst = []

    # problem = NSE(J, F)
    # solver = NewtonSolver()
    # problem = NonlinearVariationalProblem(F, w, bcs, J=J)
    # solver = NonlinearVariationalSolver(problem)
    # solver.parameters["newton_solver"]["linear_solver"] = "lu"
    # solver.parameters["newton_solver"]["convergence_criterion"] = "residual"  # "incremental"
    # solver.parameters["newton_solver"]["relative_tolerance"] = 1e-20
    # solver.parameters["newton_solver"]["absolute_tolerance"] = atol
    # solver.parameters["newton_solver"]["maximum_iterations"] = maxit
    # solver.parameters["newton_solver"]["error_on_nonconvergence"] = False
    # solver.solve()

    # dw = Function(W)
    # bcs_dw = homogenize(bcs)
    bcs_dw = [
        # DirichletBC(W.sub(0), zero, "on_boundary"),
        DirichletBC(W.sub(0), zero, bnds, 1),
        DirichletBC(W.sub(0), zero, bnds, 3),
        DirichletBC(W.sub(0).sub(1), c0, bnds, 4),
    ]

    aitken = False
    a1 = 1.
    dwk = Function(W)
    dwk.vector()[:] += 1.
    Rq = div(u)*q*dx
    for it in range(maxit):
        Aj = assemble(J)
        bf = assemble(F)
        b = assemble(F)
        [bc.apply(b) for bc in bcs]  # FIXME ??????
        [bc.apply(Aj, bf) for bc in bcs_dw]
        # Aj, bf = assemble_system(J, F, bcs_dw)

        resq = assemble(Rq)
        print(norm(resq, 'l2'))

        dunorm = dwk.vector().norm('l2')/w.vector().norm('l2')
        dunorm_lst.append(dunorm)
        resnorm_inf = bf.norm('linf')
        resnorm_inf_lst.append(resnorm_inf)
        resnorm_l2 = bf.norm('l2')
        resnorm_l2_lst.append(resnorm_l2)
        enorm = np.inner(b.array(), w.vector().array())
        print('{1}\t relax: {a:.2g}\tresid: {0:.4e}\t du: {2:.4e}\t'
              'En: {E:g}'.
              format(resnorm_l2, it, dunorm, a=a1, E=enorm))
        if resnorm_l2 < atol or dunorm < 1e-20:
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

    return w


def loop():
    from itertools import product
    Re = [100, 1000, 2400, 3600]
    metds = [2, 3, 4, 5]

    # for R in Re:
    #     for met in metds:
    cases = []
    for R, met in product(Re, metds):
        tmp = run(met, R)
        cases.append(tmp)

    return cases, Re

if __name__ == '__main__':
    _ = run(5, 2000)
