''' Gamma slip BC for Navier-Stokes equations
    author: David Nolte
    email: dnolte@dim.uchile.cl
'''
# TODO: PCFieldSplitSetBlockSize -> sub blocks? ->
# https://www.mcs.anl.gov/petsc/petsc-current/src/ksp/ksp/examples/tutorials/ex43.c.html,
# l. 1430

from dolfin import *
# petsc4py for petsc fieldsplit block preconditioner
from petsc4py import PETSc
import numpy as np
import matplotlib.pyplot as plt
# import scipy.linalg as la
# from sympy.utilities.codegen import ccode
from functions.geom import *
from mpi4py import MPI

import shutil
import os
import sys

from functions.utils import on_cluster


def _petsc_args():
    """
            --petsc.pc_type fieldsplit
            --petsc.pc_fieldsplit_detect_saddle_point
            --petsc.pc_fieldsplit_type schur
            --petsc.pc_fieldsplit_schur_factorization_type diag
            --petsc.pc_fieldsplit_schur_precondition user

            --petsc.fieldsplit_0_ksp_type richardson
            --petsc.fieldsplit_0_ksp_max_it 1
            --petsc.fieldsplit_0_pc_type lu
            --petsc.fieldsplit_0_pc_factor_mat_solver_package mumps
            --petsc.fieldsplit_0_mg_levels_ksp_type gmres
            --petsc.fieldsplit_0_mg_levels_pc_type bjacobi
            --petsc.fieldsplit_0_mg_levels_ksp_max_it 4

            --petsc.fieldsplit_1_ksp_type bcgs
            --petsc.fieldsplit_1_ksp_monitor
            --petsc.fieldsplit_1_pc_type lu
            --petsc.fieldsplit_1_pc_factor_mat_solver_package mumps
    """
    # mpiexec -n 2 ./ex70 -nx 32 -ny 48 -ksp_type fgmres -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_fact_type lower -fieldsplit_0_ksp_type gmres -fieldsplit_0_pc_type bjacobi -fieldsplit_1_pc_type jacobi -fieldsplit_1_inner_ksp_type preonly -fieldsplit_1_inner_pc_type jacobi -fieldsplit_1_upper_ksp_type preonly -fieldsplit_1_upper_pc_type jacobi
    #     Out-of-the-box SIMPLE-type preconditioning. The major advantage
    #     is that the user neither needs to provide the approximation of
    #     the Schur complement, nor the corresponding preconditioner.
    args = [sys.argv[0]] + """
            --petsc.ksp_converged_reason
            --petsc.ksp_type fgmres
            --petsc.ksp_rtol 1.0e-8
            --petsc.ksp_monitor

            --petsc.pc_fieldsplit_schur_factorization_type diag

            --petsc.fieldsplit_0_ksp_type preonly
            --petsc.fieldsplit_0_ksp_monitor
            --petsc.fieldsplit_0_pc_type ml
            --petsc.fieldsplit_0_mg_levels_ksp_type gmres
            --petsc.fieldsplit_0_mg_levels_pc_type bjacobi
            --petsc.fieldsplit_0_mg_levels_ksp_max_it 4

            --petsc.fieldsplit_1_ksp_type preonly
            --petsc.fieldsplit_1_ksp_monitor
            --petsc.fieldsplit_1_pc_type jacobi
            """.split()
    # parameters.parse(args)
    return 0

parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "quadrature"
if on_cluster():
    parameters["form_compiler"]["cpp_optimize_flags"] = "-O3 -xHost -ipo"
else:
    # parameters['num_threads'] = 2
    parameters["form_compiler"]["cpp_optimize_flags"] = \
        "-O3 -ffast-math -march=native"


def getFunctionSpaces(mesh, elements):
    ''' get all mesh depedent functions '''
    from functions.inout import read_mesh
    mesh, _, bnds = read_mesh(mesh)

    if elements == 'TH':
        V = VectorFunctionSpace(mesh, "CG", 2)
        Q = FunctionSpace(mesh, "CG", 1)
        W = V*Q
    elif elements == 'Mini':
        P1 = VectorFunctionSpace(mesh, "CG", 1)
        B = VectorFunctionSpace(mesh, "Bubble", 3)
        Q = FunctionSpace(mesh, "CG", 1)
        V = (P1 + B)
        W = V*Q

    # ds = Measure('ds', domain=mesh, subdomain_data=bnds)

    return W, bnds


def getForms(W, bnds,
             beta=10, elements='TH', gamma=0., wall='noslip',
             uwall=0.0, bctype='nitsche', symmetric=False,
             block_assemble=False, inlet='parabola', Rref=None,
             umax=1., pin=None, mu=0.01, rho=1.0):

    ndim = W.mesh().topology().dim()

    n = FacetNormal(W.mesh())
    h = CellSize(W.mesh())

    k = Constant(mu)
    rho = Constant(rho)
    beta = Constant(beta)
    if type(gamma) in [float, int]:
        gamma = Constant(gamma)

    # boundary numbering
    # 1   inlet
    bnd_in = 1
    # 2   outlet
    # bnd_out = 2
    # 3   wall
    bnd_w = 3
    # 4   slip/symmetry
    bnd_s = 4

    zero = Constant((0,)*ndim)

    bcs = []
    # inflow velocity profile
    if inlet == 'parabola':
        # y = a*(((xmax+xmin)/2 - x)**2 -(xmax-xmin)**2/4)
        if Rref:
            bmin = -Rref
            bmax = Rref
        else:
            # min/max coords on inlet boundary
            xmin = coords[:, 0].min()
            bmin = coords[coords[:, 0] == xmin, 1].min()
            bmax = coords[coords[:, 0] == xmin, 1].max()
        ys = (bmax + bmin) / 2.
        dy = (bmax - bmin) / 2.
        G = -umax/dy**2
        if ndim == 3:
            inflow = Expression(('G*(pow(ys - x[1], 2) + pow(ys - x[2], 2) - \
                                 pow(dy, 2))', '0.0', '0.0'),
                                ys=ys, dy=dy, G=G)
        elif ndim == 2:
            inflow = Expression(('G*(pow(ys - x[1], 2) + pow(ys - x[2], 2) - \
                                 pow(dy, 2))', '0.0'), ys=ys, dy=dy, G=G)
    elif inlet == 'pressure':
        if not type(pin) == dolfin.functions.constant.Constant:
            pin = Constant(pin)

    if inlet in ['parabola', 'const'] and elements == 'Mini':
        parameters['krylov_solver']['monitor_convergence'] = False
        inflow = project(inflow, W.sub(0).collapse())

    # outflow: du/dn + pn = 0 on ds(2)

    # Define variational problem
    ds = Measure('ds', domain=W.mesh(), subdomain_data=bnds)

    F = zero

    def a(u, v):
        return inner(k*grad(u), grad(v))*dx

    def b(v, q):
        return div(v)*q*dx

    def c(u, v):
        return dot(u, dot(k*grad(v), n))

    def f(v):
        return dot(F, v)*dx

    def t(u, p):
        return dot(k*grad(u), n) - p*n

    # TODO: only reasons to have the block wise assembly separate instead of a1
    # = a00 + a10 + a01, etc, are: ease of reading and maybe deprecate blocks
    # later
    if not block_assemble:
        (u, p) = TrialFunctions(W)
        (v, q) = TestFunctions(W)

        u0 = Function(W.sub(0).collapse(), name='u0')

        # Standard form Stokes equations, Dirichlet RBs
        # choose b(u, q) positive so that a(u,u) > 0
        a1 = a(u, v) - b(v, p) + b(u, q)
        aconv = rho*dot(grad(u)*u0, v)*dx + 0.5*rho*div(u0)*dot(u, v)*dx
        L1 = f(v)

        # inflow
        if inlet == 'parabola':
            if bctype == 'dirichlet':
                bcin = DirichletBC(W.sub(0), inflow, bnds, bnd_in)
                bcs.append(bcin)
            else:
                # stress tensor boundary integral
                a1 += - dot(k*grad(u)*n, v)*ds(bnd_in) \
                    + dot(p*n, v)*ds(bnd_in)
                # Nitsche BC
                a1 += beta/h*k*dot(u, v)*ds(bnd_in)
                L1 += beta/h*k*dot(inflow, v)*ds(bnd_in)
                # positive 'balance' terms
                a1 += + dot(u, k*grad(v)*n)*ds(bnd_in) \
                    - dot(u, q*n)*ds(bnd_in)
                L1 += + dot(inflow, k*grad(v)*n)*ds(bnd_in) \
                    - dot(inflow, q*n)*ds(bnd_in)
        elif inlet == 'pressure' and pin:
            # Pressure via Neumann BCs
            # XXX ---> RHS!
            # a00 += - dot(k*grad(u)*n, v)*ds(bnd_in)
            # a01 += dot(p*n, v)*ds(bnd_in)
            L1 += - dot(pin*n, v)*ds(bnd_in)
            # Nitsche: u = inflow XXX: NOTHING
            # a00 += beta/h*k*dot(u, v)*ds(bnd_in)
            # L0 += beta/h*k*dot(inflow, v)*ds(bnd_in)
            # 'balance' terms with oposite sign -> a(u, u) > 0 positive
            # a00 += + dot(u, k*grad(v)*n)*ds(bnd_in)
            # a10 += - dot(u, q*n)*ds(bnd_in)
            # L0 += + dot(inflow, k*grad(v)*n)*ds(bnd_in)
            # L1 += - dot(inflow, q*n)*ds(bnd_in)

        if symmetric:     # assume symmetric boundary for bnd_s=4 !
            # symmetric BCs (==slip)
            a1 += - dot(k*grad(u)*n, n)*dot(v, n)*ds(bnd_s) \
                + p*dot(v, n)*ds(bnd_s)
            # Nitsche: u.n = 0
            a1 += beta/h*k*dot(u, n)*dot(v, n)*ds(bnd_s)
            # RHS -> zero
            # balance terms:
            a1 += + dot(u, n)*dot(k*grad(v)*n, n)*ds(bnd_s) \
                - dot(u, n)*q*ds(bnd_s)

        if wall == 'navierslip':
            # normal component of natural boundary integral
            a1 += - dot(k*grad(u)*n, n)*dot(v, n)*ds(bnd_w) \
                + p*dot(v, n)*ds(bnd_w)
            # tangential component (natural BC t.s.n = gamma*u.t)
            # a1 += - gamma*dot(u - dot(u, n)*n, v - dot(v, n)*n)*ds(bnd_w)
            a1 += - gamma*(dot(u, v) - dot(u, n)*dot(v, n))*ds(bnd_w)
            # Nitsche: u.n = 0
            a1 += beta/h*k*dot(u, n)*dot(v, n)*ds(bnd_w)
            # RHS -> zero
            # balance terms:
            a1 += + dot(u, n)*dot(k*grad(v)*n, n)*ds(bnd_w) \
                - dot(u, n)*q*ds(bnd_w)

        if wall == 'none':
            # "do nothing" boundary condition
            a1 += - dot(k*grad(u)*n, v)*ds(bnd_w) \
                + dot(p*n, v)*ds(bnd_w)

        if wall == 'noslip':
            if ndim == 3:
                noslip = Constant((uwall, 0.0, 0.0))
            elif ndim == 2:
                noslip = Constant((uwall, 0.0))

            if elements == 'Mini':
                parameters['krylov_solver']['monitor_convergence'] = False
                noslip = project(noslip, W.sub(0).collapse())

            if bctype == 'dirichlet':
                bc0 = DirichletBC(W.sub(0), noslip, bnds, bnd_w)
                bcs.append(bc0)
            elif bctype == 'nitsche':
                # stress tensor boundary integral
                a1 += - dot(k*grad(u)*n, v)*ds(bnd_w) \
                    + dot(p*n, v)*ds(bnd_w)
                # Nitsche: u = noslip
                a1 += beta/h*k*dot(u, v)*ds(bnd_w)
                L1 += beta/h*k*dot(noslip, v)*ds(bnd_w)
                # positive 'balance' stability terms
                a1 += + dot(u, k*grad(v)*n)*ds(bnd_w) \
                    - dot(u, q*n)*ds(bnd_w)
                L1 += + dot(noslip, k*grad(v)*n)*ds(bnd_w) \
                    - dot(noslip, q*n)*ds(bnd_w)

        tic()
        # A = assemble(a1)
        # b = assemble(L1)
        # [bc.apply(A, b) for bc in bcs]
        # print('assembly time:  %f' % toc())

    else:
        # ######### BLOCK ASSEMBLE ######### #
        # gives the same form, but split into blocks
        # NO mixed function space!
        # main purpose: inspect matrices with spy(A)
        u = TrialFunction(V)
        v = TestFunction(V)
        p = TrialFunction(Q)
        q = TestFunction(Q)

        u0 = Function(V, name='u0')

        a00 = a(u, v)
        aconv = rho*dot(grad(u)*u0, v)*dx + 0.5*rho*div(u0)*dot(u, v)*dx
        a10 = b(u, q)
        a01 = - b(v, p)
        # a11 = Constant(0)*p*q*dx

        L0 = f(v)
        L1 = Constant(0)*q*dx

        # XXX TODO: SIMPLIFY FORMS --> tt = (I-nn), etc
        # inflow
        if inlet == 'parabola':
            if bctype == 'dirichlet':
                bcin = DirichletBC(V, inflow, bnds, bnd_in)
                bcs.append(bcin)
            else:
                # stress tensor boundary integral
                a00 += - dot(k*grad(u)*n, v)*ds(bnd_in)
                a01 += dot(p*n, v)*ds(bnd_in)
                # Nitsche: u = inflow
                a00 += beta/h*k*dot(u, v)*ds(bnd_in)
                L0 += beta/h*k*dot(inflow, v)*ds(bnd_in)
                # 'balance' terms with oposite sign -> a(u, u) > 0 positive
                a00 += + dot(u, k*grad(v)*n)*ds(bnd_in)
                a10 += - dot(u, q*n)*ds(bnd_in)
                L0 += + dot(inflow, k*grad(v)*n)*ds(bnd_in)
                L1 += - dot(inflow, q*n)*ds(bnd_in)
        elif inlet == 'pressure' and pin:
            # Pressure via Neumann BCs
            # XXX ---> RHS!
            # a00 += - dot(k*grad(u)*n, v)*ds(bnd_in)
            # a01 += dot(p*n, v)*ds(bnd_in)
            L1 += - dot(pin*n, v)*ds(bnd_in)
            # Nitsche: u = inflow XXX: NOTHING
            # a00 += beta/h*k*dot(u, v)*ds(bnd_in)
            # L0 += beta/h*k*dot(inflow, v)*ds(bnd_in)
            # 'balance' terms with oposite sign -> a(u, u) > 0 positive
            # a00 += + dot(u, k*grad(v)*n)*ds(bnd_in)
            # a10 += - dot(u, q*n)*ds(bnd_in)
            # L0 += + dot(inflow, k*grad(v)*n)*ds(bnd_in)
            # L1 += - dot(inflow, q*n)*ds(bnd_in)

        if symmetric:     # assume symmetric boundary for bnd_s=4 !
            # symmetric BCs (==slip)
            # normal component of natural boundary integral
            a00 += - dot(k*grad(u)*n, n)*dot(v, n)*ds(bnd_s)
            a01 += p*dot(v, n)*ds(bnd_s)
            # Nitsche: u.n = 0
            a00 += beta/h*k*dot(u, n)*dot(v, n)*ds(bnd_s)
            # RHS -> zero
            # balance terms:
            a00 += + dot(u, n)*dot(k*grad(v)*n, n)*ds(bnd_s)
            a10 += - dot(u, n)*q*ds(bnd_s)

        if wall == 'navierslip':
            # normal component of natural boundary integral
            a00 += - dot(k*grad(u)*n, n)*dot(v, n)*ds(bnd_w)
            a01 += p*dot(v, n)*ds(bnd_w)
            # tangential component (natural BC t.s.n = gamma*u.t)
            a00 += - gamma*dot(u - dot(u, n)*n, v - dot(v, n)*n)*ds(bnd_w)
            # Nitsche: u.n = 0
            a00 += beta/h*k*dot(u, n)*dot(v, n)*ds(bnd_w)
            # RHS -> zero
            # balance terms:
            a00 += + dot(u, n)*dot(k*grad(v)*n, n)*ds(bnd_w)
            a10 += - dot(u, n)*q*ds(bnd_w)

        if wall == 'none':
            # "do nothing" boundary condition
            a00 += - dot(k*grad(u)*n, v)*ds(bnd_w)
            a01 += dot(p*n, v)*ds(bnd_w)

        if wall == 'noslip':
            if ndim == 3:
                noslip = Constant((uwall, 0.0, 0.0))
            elif ndim == 2:
                noslip = Constant((uwall, 0.0))

            if elements == 'Mini':
                parameters['krylov_solver']['monitor_convergence'] = False
                noslip = project(noslip, V)

            if bctype == 'dirichlet':
                bc0 = DirichletBC(V, noslip, bnds, bnd_w)
                bcs.append(bc0)
            elif bctype == 'nitsche':
                # stress tensor boundary integral
                a00 += - dot(k*grad(u)*n, v)*ds(bnd_w)
                a01 += dot(p*n, v)*ds(bnd_w)
                # Nitsche: u = noslip
                a00 += beta/h*k*dot(u, v)*ds(bnd_w)
                L0 += beta/h*k*dot(noslip, v)*ds(bnd_w)
                # skew symmetric 'balance' terms
                a00 += + dot(u, k*grad(v)*n)*ds(bnd_w)
                a10 += - dot(u, q*n)*ds(bnd_w)
                L0 += + dot(noslip, k*grad(v)*n)*ds(bnd_w)
                L1 += -dot(noslip, q*n)*ds(bnd_w)

        # A00 = assemble(a00)
        # B = assemble(a10)
        # BT = assemble(a01)
        # b0 = assemble(L0)
        # b1 = assemble(L1)

        # XXX does this work?
        # [bc.apply(A00, B, BT, b0, b1) for bc in bcs]

        # A = (A00, B, BT)
        # b = (b0, b1)
        # W = (V, Q)

    return a1, aconv, L1, bcs


def assemble_stokes(a, L, bcs):
    A = assemble(a)
    b = assemble(L)
    [bc.apply(A, b) for bc in bcs]

    return A, b


def assemble_conv(aconv, u0, bcs):
    # extract u0 function from form
    # NAME NEEDS TO BE SET IN FUNCTION(V) DEF!
    coef = aconv.coefficients()
    for c in coef:
        if str(c) == 'u0':
            c.assign(u0)
            break

    K = assemble(aconv)
    [bc.apply(K) for bc in bcs]

    return K


def solveLS(A, b, W, solver='mumps', plots=False):
    ''' direct solver '''
    w1 = Function(W)
    solve(A, w1.vector(), b, solver)

    # Split the mixed solution
    (u1, p1) = w1.split(deepcopy=True)

    # unorm = u1.vector().norm("l2")
    # pnorm = p1.vector().norm("l2")

    # if MPI.COMM_WORLD.rank == 0:
    #     print("Norm of velocity coefficient vector: %.6g" % unorm)
    #     print("Norm of pressure coefficient vector: %.6g" % pnorm)
    #     print("DOFS: %d" % W.dim())

    if plots:
        plot(u1, title='velocity, dirichlet')
        plot(p1, title='pressure, dirichlet')

    return u1, p1


def __solveLS_numpy(A, b, W):
    A = sp.csr_matrix(A)

    # invert A
    Ainv = sl.inv(A)

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

            pc.setFieldSplitType(PETSc.PC.CompositeType.SCHUR)  # 0: additive (Jacobi)

            #                          1: multiplicative (Gauss-Seidel)
            #                          2: symmetric_multiplicative (symGS)
            #                          3: schur
            #                          see PETSc manual p. 92
            # https://www.mcs.anl.gov/petsc/petsc-3.7/docs/manualpages/PC/PCFIELDSPLIT.html#PCFIELDSPLIT
            # pc.setFieldSplitSchurFactType(PETSc.PC.SchurFactType.FULL)    # <diag,lower,upper,full>
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
            # # https://www.mcs.anl.gov/petsc/petsc-3.7/docs/manualpages/PC/PCFieldSplitSetSchurFactType.html#PCFieldSplitSetSchurFactType
            # Sp = P.getSubMatrix(is1, is1)
            # pc.setFieldSplitSchurPreType(3, Sp)
            # # == PETSc.PC.SchurPreType.USER or A11, UPPER, FULL...
            # # <----------
            # # subksps = pc.getFieldSplitSubKSP()   # CRASHES WITH SEGFAULT, SET
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


def uwall(umax, R, y):
    return -umax/R**2*(y**2 - R**2)


def solve_aorta_ref(mesh_reflvl=0):
    set_log_level(ERROR)
    Rref = 1.0

    mu = 0.01
    rho = 1.0
    beta = 20.
    umax = 2.

    elem = 'Mini'

    mesh_file_ref = 'coarc2d_f0.6_ref_r{0:d}.h5'.format(mesh_reflvl)
    mesh_ref = prep_mesh(mesh_file_ref)

    tic()

    W, bnds = getFunctionSpaces(mesh_ref, elem)
    # zero = Constant((0,)*W.mesh().topology().dim())
    # u0 = project(zero, W.sub(0).collapse())
    a, aconv, L, bcs = getForms(W, bnds,
                                beta=beta,
                                elements=elem,
                                gamma=0.,
                                wall='noslip',
                                uwall=0.0,
                                bctype='nitsche',
                                symmetric=True,
                                inlet='parabola',
                                Rref=Rref,
                                umax=umax,
                                mu=mu,
                                rho=rho
                                )

    A, b = assemble_stokes(a, L, bcs)
    u0, p0 = solveLS(A, b, W, 'mumps', plots=False)
    K = assemble_conv(aconv, u0, bcs)
    M = A + K
    uref, pref = solveLS(M, b, W, 'mumps', plots=True)

    # pressure jump
    ds = Measure('ds', domain=W.mesh(), subdomain_data=bnds)
    dPref = assemble(pref*ds(1) - pref*ds(2))

    # f1 = File("results/uref_estim.pvd")
    # f1 << uref
    # f1 = File("results/pref_estim.pvd")
    # f1 << pref

    unorm = norm(uref, "l2")
    pnorm = norm(pref, "l2")

    if MPI.COMM_WORLD.rank == 0:
        print('Pressure jump dP_ref: %f' % dPref)
        print("DOFS: %d" % W.dim())
        print("L2 norm velocity:\t %.6g" % unorm)
        print("L2 norm pressure:\t %.6g" % pnorm)
        print('time elapsed:\t %fs' % toc())

    if on_cluster():
        # delete mesh from /dev/shm when done
        os.remove(mesh_ref)

    return uref, pref, dPref


if __name__ == '__main__':

    solve_aorta_ref(0)
