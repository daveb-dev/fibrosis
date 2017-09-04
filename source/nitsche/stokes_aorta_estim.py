''' 3D pipe flow with Navier-Slip (Robin type) boundary conditions
    author: David Nolte
    email: dnolte@dim.uchile.cl

    TODO: check dependence on beta (Nitsche parameter) and alternative
    formulations
'''
# TODO: PCFieldSplitSetBlockSize -> sub blocks? ->
# https://www.mcs.anl.gov/petsc/petsc-current/src/ksp/ksp/examples/tutorials/ex43.c.html,
# l. 1430

# TODO: CHECK UNITS --> CGS

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

    ds = Measure('ds', domain=mesh, subdomain_data=bnds)
    n = FacetNormal(mesh)
    h = CellSize(mesh)

    coords = mesh.coordinates()

    return W, bnds, n, h, ds, coords


def getForms(mesh, beta=10, solver='lu', elements='TH', gamma=0.0,
             wall='noslip', uwall=0.0, bctype='nitsche', symmetric=True,
             block_assemble=False, plots=False, inlet='parabola', Rref=None,
             umax=1., pin=None, mu=0.01):
    ''' line_profiler: most time consuming components:
            read_mesh (10%)
            Mini-Elements (40%)
            project BC (noslip, Mini) (25%)
            assemble(a1)    (23%)

            --> re-use function spaces, assemble only variable parts (check
            impact!)
    '''

    pass


def getLS(W, bnds, n, h, ds, coords=None, beta=10, solver='lu', elements='TH',
          gamma=0., prec='', wall='noslip', uwall=0.0, bctype='nitsche',
          symmetric=False, block_assemble=False, plots=False, inlet='parabola',
          Rref=None, umax=1., pin=None, mu=0.01):

    ndim = W.mesh().topology().dim()

    k = Constant(mu)
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
        # E = interpolate(inflow, V)
        # plot(E)
    elif inlet == 'const':
        if ndim == 3:
            inflow = Constant((umax, 0.0, 0.0))
        elif ndim == 2:
            inflow = Constant((umax, 0.0))

    elif inlet == 'pressure':
        if not type(pin) == dolfin.functions.constant.Constant:
            pin = Constant(pin)

    if inlet in ['parabola', 'const'] and elements == 'Mini':
        parameters['krylov_solver']['monitor_convergence'] = False
        inflow = project(inflow, W.sub(0).collapse())

    # outflow: du/dn + pn = 0 on ds(2)

    # Define variational problem

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

        # Standard form Stokes equations, Dirichlet RBs
        # choose b(u, q) positive so that a(u,u) > 0
        a1 = a(u, v) - b(v, p) + b(u, q)
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
        A = assemble(a1)
        b = assemble(L1)
        [bc.apply(A, b) for bc in bcs]
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

        a00 = a(u, v)
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

        A00 = assemble(a00)
        B = assemble(a10)
        BT = assemble(a01)
        b0 = assemble(L0)
        b1 = assemble(L1)

        # XXX does this work?
        [bc.apply(A00, B, BT, b0, b1) for bc in bcs]

        A = (A00, B, BT)
        b = (b0, b1)
        W = (V, Q)

    return A, b, W


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


def solve_aorta(lvl=1):

    set_log_level(ERROR)
    # R2 = 1.0

    mu = 0.01
    # R1 = 1.0
    # dR = R2 - R1
    # gamma1 = -2.*mu*R1/(2*R1*dR + dR**2)
    # gamma2 = Expression('2.0*mu*x[1]/(x[1]*x[1] - R*R)', mu=mu, R=R2)

    beta = 20.
    # umax = 2.

    elem = 'TH'

    mesh_reflvl = lvl

    mesh_file_ref = 'coarc2d_f0.6_d-0.2_r%d.h5' % mesh_reflvl
    # mesh_file_gm = 'coarc2d_f0.6_r%d.h5' % mesh_reflvl
    mesh_ref = prep_mesh(mesh_file_ref)
    # mesh_gm = prep_mesh(mesh_file_gm)

    tic()

    W, bnds, n, h, ds, _ = getFunctionSpaces(mesh_ref, elem)
    A, b, W = getLS(W, bnds, n, h, ds,
                    wall='noslip',
                    uwall=0.0,
                    bctype='nitsche',
                    beta=beta,              # Nitsche only
                    gamma=None,            # navier-slip only
                    elements=elem,
                    symmetric=True,
                    inlet='pressure',  # parabola, pressure
                    pin=1.0,
                    Rref=None,        # true reference radius inlet
                    umax=None,      # inlet velocity
                    mu=mu
                    )
    uref, pref = solveLS(A, b, W, 'mumps', plots=True)

    # pressure jump
    dPref = assemble(pref*ds(1) - pref*ds(2))

    # f1 = File("results/uref_estim.pvd")
    # f1 << uref
    # f1 = File("results/pref_estim.pvd")
    # f1 << pref

    unorm = uref.vector().norm("l2")
    pnorm = pref.vector().norm("l2")

    if MPI.COMM_WORLD.rank == 0:
        print('Pressure jump dP_ref: %f' % dPref)
        print("DOFS: %d" % W.dim())
        print("L2 norm velocity:\t %.6g" % unorm)
        print("L2 norm pressure:\t %.6g" % pnorm)
        print('time elapsed:\t %fs' % toc())

    if on_cluster():
        # delete mesh from /dev/shm when done
        os.remove(mesh)

    return uref, pref, dPref


def STEint(u0, W, mu):
    ndim = W.mesh().topology().dim()

    (w, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)

    zero = Constant((0,)*ndim)
    noslip = project(zero, W.sub(0).collapse())
    bc = DirichletBC(W.sub(0), noslip, 'on_boundary')

    a = inner(grad(w), grad(v))*dx \
        - p*div(v)*dx + div(w)*q*dx

    L = - mu*inner(grad(u0), grad(v))*dx

    A, b = assemble_system(a, L, bc)  # , A_tensor=A_ste, b_tensor=b_ste)

    w1 = Function(W)
    solve(A, w1.vector(), b, 'mumps')

    _, p_est = w1.split(deepcopy=True)

    return p_est


def aorta_ref_STEint(mesh_reflvl=0):
    mu = 0.01
    R1 = 1.0

    beta = 20.
    umax = 2.

    elem = 'Mini'

    mesh_file_ref = 'coarc2d_f0.6_ref_r%d.h5' % mesh_reflvl
    mesh_ref = prep_mesh(mesh_file_ref)

    tic()

    W, bnds, n, h, ds, _ = getFunctionSpaces(mesh_ref, elem)
    A, b, W = getLS(W, bnds, n, h, ds,
                    wall='noslip',
                    uwall=0.0,
                    bctype='nitsche',
                    beta=beta,              # Nitsche only
                    gamma=None,            # navier-slip only
                    elements=elem,
                    symmetric=True,
                    inlet='parabola',  # parabola, pressure
                    pin=None,
                    Rref=R1,        # true reference radius inlet
                    umax=umax,      # inlet velocity
                    mu=mu
                    )
    uref, pref = solveLS(A, b, W, 'mumps', plots=False)

    # pressure jump
    dPref = assemble(pref*ds(1) - pref*ds(2))

    p_est = STEint(uref, W, mu)

    plot(p_est)

    dP = assemble(p_est*ds(1) - p_est*ds(2))

    e_dP = abs(dP - dPref)/dPref

    print('dP ref: \t {0:g}'.format(dPref))
    print('dP STEi:\t {0:g}'.format(dP))
    print('rel err:\t {0:g}'.format(e_dP))

    if on_cluster():
        # delete mesh from /dev/shm when done
        os.remove(mesh)

    return e_dP


def aorta_narrowed_STEint(mesh_reflvl=0):
    set_log_level(ERROR)
    mu = 0.01
    R1 = 1.0

    beta = 20.
    umax = 2.

    dRrange = [0.1, 0.15, 0.2]

    elem = 'Mini'

    mesh_file_ref = 'coarc2d_f0.6_ref_r%d.h5' % mesh_reflvl
    mesh_ref = prep_mesh(mesh_file_ref)
    mesh_file_gm = 'coarc2d_f0.6_d{0:g}_r{1:d}.h5'  # .format(dR, mesh_reflvl)

    tic()

    W, bnds, n, h, ds, _ = getFunctionSpaces(mesh_ref, elem)
    A, b, W = getLS(W, bnds, n, h, ds,
                    wall='noslip',
                    uwall=0.0,
                    bctype='nitsche',
                    beta=beta,              # Nitsche only
                    gamma=None,            # navier-slip only
                    elements=elem,
                    symmetric=True,
                    inlet='parabola',  # parabola, pressure
                    pin=None,
                    Rref=R1,        # true reference radius inlet
                    umax=umax,      # inlet velocity
                    mu=mu
                    )
    uref, pref = solveLS(A, b, W, 'mumps', plots=False)

    # pressure jump
    dPref = assemble(pref*ds(1) - pref*ds(2))

    p_est = STEint(uref, W, mu)

    # plot(p_est)

    dPest = assemble(p_est*ds(1) - p_est*ds(2))

    e_dP_ref = abs(dPest - dPref)/dPref
    print('reference mesh:')
    print('dP ref: \t {0:g}'.format(dPref))
    print('dP STEi:\t {0:g}'.format(dPest))
    print('rel err:\t {0:g}'.format(e_dP_ref))

    if elem == 'Mini':
        uref = interpolate(uref, VectorFunctionSpace(W.mesh(), 'CG', 1))

    e_dP = []
    e_dPi = []
    for i, dR in enumerate(dRrange):
        tic()

        mesh_gm = prep_mesh(mesh_file_gm.format(dR, mesh_reflvl))
        W, bnds, n, h, ds, _ = getFunctionSpaces(mesh_gm, elem)

        if elem == 'Mini':
            # special treatment for enriched elements
            # https://bitbucket.org/fenics-project/dolfin/issues/489/wrong-interpolation-for-enriched-element#comment-16480392
            ''' This effectively disables interpolation to enriched
            space, projection from enriched to another mesh and using
            expressions and constants in Dirichlet BCs on enriched
            space. The rest (i.e.  projection to enriched,
            interpolation from enriched to non-enriched on the same
            mesh and Functions in Dirichlet values on enriched space)
            should be working correctly.
            '''
            parameters['krylov_solver']['monitor_convergence'] = False
            V = W.sub(0).collapse()
            urefi = project(uref, V)
        else:
            urefi = interpolate(uref, W.sub(0).collapse())
        prefi = interpolate(pref, W.sub(1).collapse())


        # estimate pressure (watch out: sign of u changed above!)
        p_est = STEint(urefi, W, mu)
        # pressure jump
        dPrefi = assemble(prefi*ds(1) - prefi*ds(2))

        dPest = assemble(p_est*ds(1) - p_est*ds(2))
        e_dPi.append(abs(dPest - dPrefi)/dPrefi)
        e_dP.append(abs(dPest - dPref)/dPref)

        print('\ndR/R: {0}'.format(dR/R1))
        print('dP interp:\t {0:g}'.format(dPrefi))
        print('dP est: \t {0:g}'.format(dPest))
        print('dP ref: \t {0:g}'.format(dPref))
        print('dP err: \t {0:g}'.format(e_dP[-1]))
        print('dP err wrt int:\t {0:g}'.format(e_dPi[-1]))

        if on_cluster():
            # delete mesh from /dev/shm when done
            os.remove(mesh_gm)

    if on_cluster():
        # delete mesh from /dev/shm when done
        os.remove(mesh_ref)

    return e_dP_ref, e_dP, e_dPi


def aorta_narrowed_STEint_measrmt():
    set_log_level(ERROR)

    H = 0.15
    elem = 'Mini'
    elem_ref = 'TH'
    mu = 0.01
    Rref = 1.0
    beta = 20.

    dRrange = [0.1, 0.2]

    mesh_file_ref = 'coarc2d_f0.6_ref_r{0:d}.h5'.format(0)
    mesh_file_meas = 'coarc2d_f0.6_d{0:g}_h{1:g}.h5'
    mesh_ref = prep_mesh(mesh_file_ref)

    tic()
    umax = 2.0
    W, bnds, n, h, ds, _ = getFunctionSpaces(mesh_ref, elem)
    A, b, W = getLS(W, bnds, n, h, ds,
                    wall='noslip',
                    uwall=0.0,
                    bctype='nitsche',
                    beta=beta,              # Nitsche only
                    gamma=None,            # navier-slip only
                    elements=elem_ref,
                    symmetric=True,
                    inlet='parabola',  # parabola, pressure
                    pin=None,
                    Rref=Rref,        # true reference radius inlet
                    umax=umax,      # inlet velocity
                    mu=mu
                    )
    uref, pref = solveLS(A, b, W, 'mumps', plots=False)

    # pressure jump
    dPref = assemble(pref*ds(1) - pref*ds(2))

    p_est = STEint(uref, W, mu)

    # plot(p_est)

    dPest = assemble(p_est*ds(1) - p_est*ds(2))

    e_dP_ref = abs(dPest - dPref)/dPref
    print('reference mesh:')
    print('dP ref: \t {0:g}'.format(dPref))
    print('dP STEi:\t {0:g}'.format(dPest))
    print('rel err:\t {0:g}'.format(e_dP_ref))

    if elem == 'Mini':
        uref = interpolate(uref, VectorFunctionSpace(W.mesh(), 'CG', 1))

    e_dP = []
    e_dPmeas = []
    for i, dR in enumerate(dRrange):
        tic()

        mesh_meas = prep_mesh(mesh_file_meas.format(dR, H))
        W, _, _, _, ds, _ = getFunctionSpaces(mesh_meas, elem)

        # interpolate uref(P2) to umeas(P1) on mesh_gm
        Vmeas = VectorFunctionSpace(W.mesh(), 'CG', 1)
        umeas = interpolate(uref, Vmeas)
        pmeas = interpolate(pref, W.sub(1).collapse())

        # estimate pressure (watch out: sign of u changed above!)
        p_est = STEint(umeas, W, mu)
        # pressure jump
        dPmeas = assemble(pmeas*ds(1) - pmeas*ds(2))

        dPest = assemble(p_est*ds(1) - p_est*ds(2))
        e_dPmeas.append(abs(dPest - dPmeas)/dPmeas)
        e_dP.append(abs(dPest - dPref)/dPref)

        print('\ndR/R: {0}'.format(dR/Rref))
        print('dP meas:\t {0:g}'.format(dPmeas))
        print('dP STEi: \t {0:g}'.format(dPest))
        print('dP ref: \t {0:g}'.format(dPref))
        print('e(dP) w.r.t. ref: \t {0:g}'.format(e_dP[-1]))
        print('e(dP) w.r.t. meas:\t {0:g}'.format(e_dPmeas[-1]))

        if on_cluster():
            # delete mesh from /dev/shm when done
            os.remove(mesh_meas)

    if on_cluster():
        # delete mesh from /dev/shm when done
        os.remove(mesh_ref)

    return e_dP_ref, e_dP, e_dPmeas


def aorta_narrowed_est_noslip():
    ''' simulate over parameter grid [gamma1, gamma2, Pin]
        reference solution: big mesh, no-slip, umax = 2.0
        parameters: gamma1*a, a = (0..2)
                    gamma2*a, a = (0..2)
                    pin*a, a = (0..2), pin = 0.45
    '''

    # geometry/setup
    set_log_level(ERROR)
    mesh_reflvl = 1
    mesh_reflvl = 0
    mesh_file_ref = 'coarc2d_f0.6_ref_r{0:d}.h5'.format(mesh_reflvl)
    mesh_file_gm = 'coarc2d_f0.6_d{0:g}_r{1:d}.h5'  # .format(dR, mesh_reflvl)
    mesh_ref = prep_mesh(mesh_file_ref)
    # mesh_gm = prep_mesh(mesh_file_gm)

    elem = 'Mini'
    mu = 0.01
    # R1 = 1.0
    # R2 = 1.0
    beta = 20.

    # define parameter grid
    dRrange = [0.1]  # , 0.15]  # [0.1, 0.15, 0.2]

    pbase = 0.45
    # pbase = 1.0
    pinrange = pbase*np.arange(0.5, 2.5, 0.2)
    # pinrange = [pbase]

    # Calculate reference solution
    # umax = 2.
    W, bnds, n, h, ds, _ = getFunctionSpaces(mesh_ref, elem)

    A, b, W = getLS(W, bnds, n, h, ds,
                    wall='noslip',
                    uwall=0.0,
                    bctype='nitsche',
                    beta=beta,              # Nitsche only
                    gamma=None,            # navier-slip only
                    elements=elem,
                    symmetric=True,
                    inlet='pressure',  # parabola, const
                    Rref=None,        # true reference radius inlet
                    umax=None,      # inlet velocity
                    pin=0.46,
                    mu=mu
                    )
    uref, pref = solveLS(A, b, W, 'mumps', plots=False)

    if elem == 'Mini':
        # interpolate solution from enriched elements to standard elements
        V = VectorFunctionSpace(W.mesh(), "CG", 1)
        uref = interpolate(uref, V)
        del(V)

    dPref = assemble(pref*ds(1) - pref*ds(2))
    print('Pressure jump dP_ref: %f' % dPref)

    # Loop over parameter grid
    e_u_inf = []
    e_u_l2 = []
    e_p_inf = []
    e_p_l2 = []
    e_dP = []
    N = len(pinrange)*len(dRrange)
    k = 0
    for dR in dRrange:
        i = 0
        mesh_gm = prep_mesh(mesh_file_gm.format(dR, mesh_reflvl))

        tic()
        W, bnds, n, h, ds, _ = getFunctionSpaces(mesh_gm, elem)
        print('building function spaces:  %fs' % toc())

        for pin in pinrange:

            tic()
            A, b, W = getLS(W, bnds, n, h, ds,
                            wall='noslip',
                            uwall=0.0,
                            # uwall=uwall(umax, R2, R1),  # noslip only
                            bctype='nitsche',
                            beta=beta,              # Nitsche only
                            elements=elem,
                            symmetric=True,
                            inlet='pressure',  # parabola, pressure
                            pin=pin,            # inlet pressure
                            Rref=None,        # true reference radius inlet
                            umax=None,      # inlet velocity
                            mu=mu
                            )
            u, p = solveLS(A, b, W, 'mumps', plots=False)
            t_solv = toc()

            # compute error w.r.t. reference solution uref, pref

            # interpolate/project uref,pref to W on first iteration
            if i == 0:
                if elem == 'Mini':
                    # special treatment for enriched elements
                    # https://bitbucket.org/fenics-project/dolfin/issues/489/wrong-interpolation-for-enriched-element#comment-16480392
                    ''' This effectively disables interpolation to enriched
                    space, projection from enriched to another mesh and using
                    expressions and constants in Dirichlet BCs on enriched
                    space. The rest (i.e.  projection to enriched,
                    interpolation from enriched to non-enriched on the same
                    mesh and Functions in Dirichlet values on enriched space)
                    should be working correctly.
                    '''
                    parameters['krylov_solver']['monitor_convergence'] = False
                    # V = W.sub(0).collapse()
                    # urefi = project(uref, V)
                    urefi = project(uref, u.function_space())
                else:
                    # urefi = interpolate(uref, W.sub(0).collapse())
                    urefi = interpolate(uref, u.function_space())
                prefi = interpolate(pref, p.function_space())

                unorm_ref = norm(urefi.vector(), 'l2')
                pnorm_ref = norm(prefi.vector(), 'l2')
                unorm_ref_inf = norm(urefi.vector(), 'linf')
                pnorm_ref_inf = norm(prefi.vector(), 'linf')

            # du = Function(V)
            # du.assign(interpolate(u, V) - uref)
            # print(norm(du, 'l2')/norm(uref, 'l2'))
            # print(norm(u.vector() - uref.vector(), 'l2') /
            #       norm(uref.vector(), 'l2'))
            e_u_inf.append(norm(u.vector() - urefi.vector(), 'linf') /
                           unorm_ref_inf)
            e_p_inf.append(norm(p.vector() - prefi.vector(), 'linf') /
                           pnorm_ref_inf)
            e_u_l2.append(norm(u.vector() - urefi.vector(), 'l2')/unorm_ref)
            e_p_l2.append(norm(p.vector() - prefi.vector(), 'l2')/pnorm_ref)
            dP = assemble(p*ds(1) - p*ds(2))
            e_dP.append(abs(dP - dPref)/dPref)

            if MPI.COMM_WORLD.rank == 0:
                print('run # %i/%i \t d = %g' % ((i+1)*(k+1), N, dR))
                print('assembly and solver time:  %.4fs' % t_solv)
                print('e(dP):   \t %f' % e_dP[-1])
                print('e(u)_inf:\t %f' % e_u_inf[-1])
                print('e(u)_l2: \t %f' % e_u_l2[-1])
                print('e(p)_inf:\t %f' % e_p_inf[-1])
                print('e(p)_l2: \t %f' % e_p_l2[-1])

            i += 1
        k += 1
        if on_cluster():
            os.remove(mesh_gm)

    if on_cluster():
        # delete mesh from /dev/shm when done
        os.remove(mesh_ref)

    n1 = len(pinrange)
    n2 = len(dRrange)
    shape = (n2, n1)

    return (pinrange, dRrange,
            np.reshape(np.array(e_u_inf), shape),
            np.reshape(np.array(e_u_l2), shape),
            np.reshape(np.array(e_p_inf), shape),
            np.reshape(np.array(e_p_inf), shape),
            np.reshape(np.array(e_dP), shape)), (uref, u)


def aorta_narrowed_est_noslip_popt(recalc=False, mesh_reflvl=0):
    ''' simulate over parameter grid [gamma1, gamma2, Pin]
        reference solution: big mesh, no-slip, umax = 2.0
        parameters: gamma1*a, a = (0..2)
                    gamma2*a, a = (0..2)
        pressure from optimal_coefficient()
    '''

    # geometry/setup
    set_log_level(ERROR)
    mesh_file_ref = 'coarc2d_f0.6_ref_r{0:d}.h5'.format(mesh_reflvl)
    mesh_file_gm = 'coarc2d_f0.6_d{0:g}_r{1:d}.h5'  # .format(dR, mesh_reflvl)
    mesh_ref = prep_mesh(mesh_file_ref)
    # mesh_gm = prep_mesh(mesh_file_gm)

    elem = 'Mini'
    mu = 0.01
    # R1 = 1.0
    # R2 = 1.0
    beta = 20.

    # define parameter grid
    dRrange = [0.1, 0.15, 0.2]

    pbase = 1.0

    # Calculate reference solution
    # umax = 2.
    W, bnds, n, h, ds, _ = getFunctionSpaces(mesh_ref, elem)

    A, b, W = getLS(W, bnds, n, h, ds,
                    wall='noslip',
                    uwall=0.0,
                    bctype='nitsche',
                    beta=beta,              # Nitsche only
                    gamma=None,            # navier-slip only
                    elements=elem,
                    symmetric=True,
                    inlet='pressure',  # parabola, const
                    Rref=None,        # true reference radius inlet
                    umax=None,      # inlet velocity
                    pin=0.46,
                    mu=mu
                    )
    uref, pref = solveLS(A, b, W, 'mumps', plots=False)

    if elem == 'Mini':
        # interpolate solution from enriched elements to standard elements
        V = VectorFunctionSpace(W.mesh(), "CG", 1)
        uref = interpolate(uref, V)
        del(V)

    dPref = assemble(pref*ds(1) - pref*ds(2))
    print('Pressure jump dP_ref: %f' % dPref)

    # Loop over parameter grid
    e_u_inf = []
    e_u_l2 = []
    e_p_inf = []
    e_p_l2 = []
    e_dP = []
    N = len(dRrange)
    k = 0
    for dR in dRrange:
        mesh_gm = prep_mesh(mesh_file_gm.format(dR, mesh_reflvl))

        tic()
        W, bnds, n, h, ds, _ = getFunctionSpaces(mesh_gm, elem)
        print('building function spaces:  %fs' % toc())

        tic()
        A, b, W = getLS(W, bnds, n, h, ds,
                        wall='noslip',
                        uwall=0.0,
                        # uwall=uwall(umax, R2, R1),  # noslip only
                        bctype='nitsche',
                        beta=beta,              # Nitsche only
                        elements=elem,
                        symmetric=True,
                        inlet='pressure',  # parabola, pressure
                        pin=pbase,            # inlet pressure
                        Rref=None,        # true reference radius inlet
                        umax=None,      # inlet velocity
                        mu=mu
                        )
        u, p = solveLS(A, b, W, 'mumps', plots=False)
        t_solv = toc()

        # compute error w.r.t. reference solution uref, pref

        # interpolate/project uref,pref to W on first iteration
        if elem == 'Mini':
            # special treatment for enriched elements
            # https://bitbucket.org/fenics-project/dolfin/issues/489/wrong-interpolation-for-enriched-element#comment-16480392
            ''' This effectively disables interpolation to enriched
            space, projection from enriched to another mesh and using
            expressions and constants in Dirichlet BCs on enriched
            space. The rest (i.e.  projection to enriched,
            interpolation from enriched to non-enriched on the same
            mesh and Functions in Dirichlet values on enriched space)
            should be working correctly.
            '''
            parameters['krylov_solver']['monitor_convergence'] = False
            # V = W.sub(0).collapse()
            # urefi = project(uref, V)
            urefi = project(uref, u.function_space())
        else:
            # urefi = interpolate(uref, W.sub(0).collapse())
            urefi = interpolate(uref, u.function_space())
        prefi = interpolate(pref, p.function_space())

        unorm_ref = norm(urefi.vector(), 'l2')
        pnorm_ref = norm(prefi.vector(), 'l2')
        unorm_ref_inf = norm(urefi.vector(), 'linf')
        pnorm_ref_inf = norm(prefi.vector(), 'linf')

        # Estimate optimal pressure from (uref, u)
        a_opt = optimal_coefficient(urefi, u)

        if recalc:
            A, b, W = getLS(W, bnds, n, h, ds,
                            wall='noslip',
                            uwall=0.0,
                            # uwall=uwall(umax, R2, R1),  # noslip only
                            bctype='nitsche',
                            beta=beta,              # Nitsche only
                            elements=elem,
                            symmetric=True,
                            inlet='pressure',  # parabola, pressure
                            pin=pbase,            # inlet pressure
                            Rref=None,        # true reference radius inlet
                            umax=None,      # inlet velocity
                            mu=mu
                            )
            u, p = solveLS(A, b, W, 'mumps', plots=False)
        else:
            u.vector()[:] *= a_opt
            p.vector()[:] *= a_opt
        # du = Function(V)
        # du.assign(interpolate(u, V) - uref)
        # print(norm(du, 'l2')/norm(uref, 'l2'))
        # print(norm(u.vector() - uref.vector(), 'l2') /
        #       norm(uref.vector(), 'l2'))
        e_u_inf.append(norm(u.vector() - urefi.vector(), 'linf') /
                       unorm_ref_inf)
        e_p_inf.append(norm(p.vector() - prefi.vector(), 'linf') /
                       pnorm_ref_inf)
        e_u_l2.append(norm(u.vector() - urefi.vector(), 'l2')/unorm_ref)
        e_p_l2.append(norm(p.vector() - prefi.vector(), 'l2')/pnorm_ref)

        dP = assemble(p*ds(1) - p*ds(2))
        e_dP.append(abs(dP - dPref)/dPref)

        if MPI.COMM_WORLD.rank == 0:
            print('run # %i/%i \t d = %g' % (k+1, N, dR))
            print('assembly and solver time:  %.4fs' % t_solv)
            print('e(dP):   \t %f' % e_dP[-1])
            print('e(u)_inf:\t %f' % e_u_inf[-1])
            print('e(u)_l2: \t %f' % e_u_l2[-1])
            print('e(p)_inf:\t %f' % e_p_inf[-1])
            print('e(p)_l2: \t %f' % e_p_l2[-1])

        k += 1
        if on_cluster():
            os.remove(mesh_gm)

    if on_cluster():
        # delete mesh from /dev/shm when done
        os.remove(mesh_ref)

    shape = (len(dRrange), 1)

    return (dRrange,
            np.reshape(np.array(e_u_inf), shape),
            np.reshape(np.array(e_u_l2), shape),
            np.reshape(np.array(e_p_inf), shape),
            np.reshape(np.array(e_p_inf), shape),
            np.reshape(np.array(e_dP), shape)), (uref, u)


def aorta_narrowed_est_noslip_inflow(mesh_reflvl=0):
    ''' simulate over parameter grid [gamma1, gamma2, Pin]
        reference solution: big mesh, no-slip, umax = 2.0
        parameters: gamma1*a, a = (0..2)
                    gamma2*a, a = (0..2)
        pressure from optimal_coefficient()
    '''

    # geometry/setup
    set_log_level(ERROR)
    mesh_file_ref = 'coarc2d_f0.6_ref_r{0:d}.h5'.format(mesh_reflvl)
    mesh_file_gm = 'coarc2d_f0.6_d{0:g}_r{1:d}.h5'  # .format(dR, mesh_reflvl)
    mesh_ref = prep_mesh(mesh_file_ref)
    # mesh_gm = prep_mesh(mesh_file_gm)

    elem = 'Mini'
    mu = 0.01
    R1 = 1.0
    # R2 = 1.0
    beta = 20.

    # define parameter grid
    dRrange = [0.1, 0.15, 0.2]

    # pbase = 1.0
    umaxbase = 2.0
    umaxrange = umaxbase*np.arange(0.5, 1.5, 0.25)

    # Calculate reference solution
    # umax = 2.
    W, bnds, n, h, ds, _ = getFunctionSpaces(mesh_ref, elem)

    A, b, W = getLS(W, bnds, n, h, ds,
                    wall='noslip',
                    uwall=0.0,
                    bctype='nitsche',
                    beta=beta,              # Nitsche only
                    gamma=None,            # navier-slip only
                    elements=elem,
                    symmetric=True,
                    inlet='parabola',  # parabola, const
                    Rref=R1,        # true reference radius inlet
                    umax=umaxbase,      # inlet velocity
                    pin=None,
                    mu=mu
                    )
    uref, pref = solveLS(A, b, W, 'mumps', plots=False)

    if elem == 'Mini':
        # interpolate solution from enriched elements to standard elements
        V = VectorFunctionSpace(W.mesh(), "CG", 1)
        uref = interpolate(uref, V)
        del(V)

    dPref = assemble(pref*ds(1) - pref*ds(2))
    print('Pressure jump dP_ref: %f' % dPref)

    # Loop over parameter grid
    e_u_inf = []
    e_u_l2 = []
    e_p_inf = []
    e_p_l2 = []
    e_dP = []
    N = len(dRrange)*len(umaxrange)
    k = 0
    for dR in dRrange:
        mesh_gm = prep_mesh(mesh_file_gm.format(dR, mesh_reflvl))

        R = R1 - dR

        tic()
        W, bnds, n, h, ds, _ = getFunctionSpaces(mesh_gm, elem)
        print('building function spaces:  %fs' % toc())

        for i, umax in enumerate(umaxrange):
            tic()
            A, b, W = getLS(W, bnds, n, h, ds,
                            wall='noslip',
                            uwall=0.0,
                            # uwall=uwall(umax, R2, R1),  # noslip only
                            bctype='nitsche',
                            beta=beta,              # Nitsche only
                            elements=elem,
                            symmetric=True,
                            inlet='parabola',  # parabola, pressure
                            pin=None,            # inlet pressure
                            Rref=R,        # true reference radius inlet
                            umax=umax,      # inlet velocity
                            mu=mu
                            )
            u, p = solveLS(A, b, W, 'mumps', plots=False)
            t_solv = toc()

            # compute error w.r.t. reference solution uref, pref

            # interpolate/project uref,pref to W on first iteration
            if elem == 'Mini':
                # special treatment for enriched elements
                # https://bitbucket.org/fenics-project/dolfin/issues/489/wrong-interpolation-for-enriched-element#comment-16480392
                ''' This effectively disables interpolation to enriched
                space, projection from enriched to another mesh and using
                expressions and constants in Dirichlet BCs on enriched
                space. The rest (i.e.  projection to enriched,
                interpolation from enriched to non-enriched on the same
                mesh and Functions in Dirichlet values on enriched space)
                should be working correctly.
                '''
                parameters['krylov_solver']['monitor_convergence'] = False
                # V = W.sub(0).collapse()
                # urefi = project(uref, V)
                urefi = project(uref, u.function_space())
            else:
                # urefi = interpolate(uref, W.sub(0).collapse())
                urefi = interpolate(uref, u.function_space())
            prefi = interpolate(pref, p.function_space())

            unorm_ref = norm(urefi.vector(), 'l2')
            pnorm_ref = norm(prefi.vector(), 'l2')
            unorm_ref_inf = norm(urefi.vector(), 'linf')
            pnorm_ref_inf = norm(prefi.vector(), 'linf')

            # Estimate optimal pressure from (uref, u)
            # a_opt = optimal_coefficient(urefi, u)
            # u.vector()[:] *= a_opt
            # p.vector()[:] *= a_opt

            # du = Function(V)
            # du.assign(interpolate(u, V) - uref)
            # print(norm(du, 'l2')/norm(uref, 'l2'))
            # print(norm(u.vector() - uref.vector(), 'l2') /
            #       norm(uref.vector(), 'l2'))
            e_u_inf.append(norm(u.vector() - urefi.vector(), 'linf') /
                           unorm_ref_inf)
            e_p_inf.append(norm(p.vector() - prefi.vector(), 'linf') /
                           pnorm_ref_inf)
            e_u_l2.append(norm(u.vector() - urefi.vector(), 'l2')/unorm_ref)
            e_p_l2.append(norm(p.vector() - prefi.vector(), 'l2')/pnorm_ref)

            dP = assemble(p*ds(1) - p*ds(2))
            e_dP.append(abs(dP - dPref)/dPref)

            if MPI.COMM_WORLD.rank == 0:
                print('run # %i/%i \t d = %g \t umax = %g' %
                      (i+1+k*len(umaxrange), N, dR, umax))
                print('assembly and solver time:  %.4fs' % t_solv)
                print('e(dP):   \t %f' % e_dP[-1])
                print('e(u)_inf:\t %f' % e_u_inf[-1])
                print('e(u)_l2: \t %f' % e_u_l2[-1])
                print('e(p)_inf:\t %f' % e_p_inf[-1])
                print('e(p)_l2: \t %f' % e_p_l2[-1])

        k += 1
        if on_cluster():
            os.remove(mesh_gm)

    if on_cluster():
        # delete mesh from /dev/shm when done
        os.remove(mesh_ref)

    shape = (len(dRrange), len(umaxrange))

    return (dRrange, umaxrange,
            np.reshape(np.array(e_u_inf), shape),
            np.reshape(np.array(e_u_l2), shape),
            np.reshape(np.array(e_p_inf), shape),
            np.reshape(np.array(e_p_inf), shape),
            np.reshape(np.array(e_dP), shape)), (uref, u)


def aorta_narrowed_est_noslip_opt_inflow(mesh_reflvl=0):
    ''' simulate over parameter grid [gamma1, gamma2, Pin]
        reference solution: big mesh, no-slip, umax = 2.0
        parameters: gamma1*a, a = (0..2)
                    gamma2*a, a = (0..2)
        pressure from optimal_coefficient()
    '''

    # geometry/setup
    set_log_level(ERROR)
    mesh_file_ref = 'coarc2d_f0.6_ref_r{0:d}.h5'.format(mesh_reflvl)
    mesh_file_gm = 'coarc2d_f0.6_d{0:g}_r{1:d}.h5'  # .format(dR, mesh_reflvl)
    mesh_ref = prep_mesh(mesh_file_ref)
    # mesh_gm = prep_mesh(mesh_file_gm)

    elem = 'Mini'
    mu = 0.01
    R1 = 1.0
    # R2 = 1.0
    beta = 20.

    # define parameter grid
    dRrange = [0.1, 0.15, 0.2]

    # pbase = 1.0
    umaxbase = 1.3

    # Calculate reference solution
    # umax = 2.
    W, bnds, n, h, ds, _ = getFunctionSpaces(mesh_ref, elem)

    A, b, W = getLS(W, bnds, n, h, ds,
                    wall='noslip',
                    uwall=0.0,
                    bctype='nitsche',
                    beta=beta,              # Nitsche only
                    gamma=None,            # navier-slip only
                    elements=elem,
                    symmetric=True,
                    inlet='parabola',  # parabola, const
                    Rref=R1,        # true reference radius inlet
                    umax=2.0,      # inlet velocity
                    pin=None,
                    mu=mu
                    )
    uref, pref = solveLS(A, b, W, 'mumps', plots=False)

    if elem == 'Mini':
        # interpolate solution from enriched elements to standard elements
        V = VectorFunctionSpace(W.mesh(), "CG", 1)
        uref = interpolate(uref, V)
        del(V)

    dPref = assemble(pref*ds(1) - pref*ds(2))
    print('Pressure jump dP_ref: %f' % dPref)

    # Loop over parameter grid
    e_u_inf = []
    e_u_l2 = []
    e_p_inf = []
    e_p_l2 = []
    e_dP = []
    umax = []
    N = len(dRrange)
    k = 0
    for dR in dRrange:
        mesh_gm = prep_mesh(mesh_file_gm.format(dR, mesh_reflvl))

        R = R1 - dR

        tic()
        W, bnds, n, h, ds, _ = getFunctionSpaces(mesh_gm, elem)
        print('building function spaces:  %fs' % toc())

        tic()
        A, b, W = getLS(W, bnds, n, h, ds,
                        wall='noslip',
                        uwall=0.0,
                        # uwall=uwall(umax, R2, R1),  # noslip only
                        bctype='nitsche',
                        beta=beta,              # Nitsche only
                        elements=elem,
                        symmetric=True,
                        inlet='parabola',  # parabola, pressure
                        pin=None,            # inlet pressure
                        Rref=R,        # true reference radius inlet
                        umax=umaxbase,      # inlet velocity
                        mu=mu
                        )
        u, p = solveLS(A, b, W, 'mumps', plots=False)
        t_solv = toc()

        # compute error w.r.t. reference solution uref, pref

        # interpolate/project uref,pref to W on first iteration
        if elem == 'Mini':
            # special treatment for enriched elements
            # https://bitbucket.org/fenics-project/dolfin/issues/489/wrong-interpolation-for-enriched-element#comment-16480392
            ''' This effectively disables interpolation to enriched
            space, projection from enriched to another mesh and using
            expressions and constants in Dirichlet BCs on enriched
            space. The rest (i.e.  projection to enriched,
            interpolation from enriched to non-enriched on the same
            mesh and Functions in Dirichlet values on enriched space)
            should be working correctly.
            '''
            parameters['krylov_solver']['monitor_convergence'] = False
            # V = W.sub(0).collapse()
            # urefi = project(uref, V)
            urefi = project(uref, u.function_space())
        else:
            # urefi = interpolate(uref, W.sub(0).collapse())
            urefi = interpolate(uref, u.function_space())
        prefi = interpolate(pref, p.function_space())

        unorm_ref = norm(urefi.vector(), 'l2')
        pnorm_ref = norm(prefi.vector(), 'l2')
        unorm_ref_inf = norm(urefi.vector(), 'linf')
        pnorm_ref_inf = norm(prefi.vector(), 'linf')

        # Estimate optimal umax from (uref, u)
        # not pressure, but works for inflow, too!
        a_opt = optimal_coefficient(urefi, u)
        u.vector()[:] *= a_opt
        p.vector()[:] *= a_opt

        umax.append(umaxbase*a_opt)

        # du = Function(V)
        # du.assign(interpolate(u, V) - uref)
        # print(norm(du, 'l2')/norm(uref, 'l2'))
        # print(norm(u.vector() - uref.vector(), 'l2') /
        #       norm(uref.vector(), 'l2'))
        e_u_inf.append(norm(u.vector() - urefi.vector(), 'linf') /
                       unorm_ref_inf)
        e_p_inf.append(norm(p.vector() - prefi.vector(), 'linf') /
                       pnorm_ref_inf)
        e_u_l2.append(norm(u.vector() - urefi.vector(), 'l2')/unorm_ref)
        e_p_l2.append(norm(p.vector() - prefi.vector(), 'l2')/pnorm_ref)

        dP = assemble(p*ds(1) - p*ds(2))
        e_dP.append(abs(dP - dPref)/dPref)

        if MPI.COMM_WORLD.rank == 0:
            print('run # %i/%i \t d = %g \t umax = %g' % (k+1, N, dR, umax[-1]))
            print('assembly and solver time:  %.4fs' % t_solv)
            print('e(dP):   \t %f' % e_dP[-1])
            print('e(u)_inf:\t %f' % e_u_inf[-1])
            print('e(u)_l2: \t %f' % e_u_l2[-1])
            print('e(p)_inf:\t %f' % e_p_inf[-1])
            print('e(p)_l2: \t %f' % e_p_l2[-1])

        k += 1
        if on_cluster():
            os.remove(mesh_gm)

    if on_cluster():
        # delete mesh from /dev/shm when done
        os.remove(mesh_ref)

    shape = (len(dRrange), 1)

    return (dRrange, umax,
            np.reshape(np.array(e_u_inf), shape),
            np.reshape(np.array(e_u_l2), shape),
            np.reshape(np.array(e_p_inf), shape),
            np.reshape(np.array(e_p_inf), shape),
            np.reshape(np.array(e_dP), shape)), (uref, u)


def aorta_narrowed_est_noslip_uopt_inflow_measrmt():
    ''' simulate over parameter grid [gamma1, gamma2, Pin]
        reference solution: big mesh, no-slip, umax = 2.0
        parameters: gamma1*a, a = (0..2)
                    gamma2*a, a = (0..2)
        pressure from optimal_coefficient()
    '''

    # geometry/setup
    set_log_level(ERROR)
    mesh_file_ref = 'coarc2d_f0.6_ref_r{0:d}.h5'.format(0)
    mesh_file_gm = 'coarc2d_f0.6_d{0:g}_h{1:g}.h5'
    # mesh_file_meas = 'coarc2d_f0.6_d{0:g}_h{1:g}.h5'

    mesh_ref = prep_mesh(mesh_file_ref)

    H = 0.15
    h1 = 0.1
    elem = 'Mini'
    elem_ref = 'TH'
    mu = 0.01
    Rref = 1.0
    beta = 20.

    # define parameter grid
    dRrange = [0.1, 0.2]

    # pbase = 1.0
    umaxbase = 1.3

    # Calculate reference solution
    # umax = 2.
    W, bnds, n, h, ds, _ = getFunctionSpaces(mesh_ref, elem)

    A, b, W = getLS(W, bnds, n, h, ds,
                    wall='noslip',
                    uwall=0.0,
                    bctype='nitsche',
                    beta=beta,              # Nitsche only
                    gamma=None,            # navier-slip only
                    elements=elem_ref,
                    symmetric=True,
                    inlet='parabola',  # parabola, const
                    Rref=Rref,        # true reference radius inlet
                    umax=2.0,      # inlet velocity
                    pin=None,
                    mu=mu
                    )
    uref, pref = solveLS(A, b, W, 'mumps', plots=False)

    if elem == 'Mini':
        # interpolate solution from enriched elements to standard elements
        V = VectorFunctionSpace(W.mesh(), "CG", 1)
        uref = interpolate(uref, V)
        del(V)

    dPref = assemble(pref*ds(1) - pref*ds(2))
    print('Pressure jump dP_ref: %f' % dPref)

    # Loop over parameter grid
    e_u_inf = []
    e_u_l2 = []
    e_p_inf = []
    e_p_l2 = []
    e_dP = []
    e_dPmeas = []
    umaxopt = []
    N = len(dRrange)
    k = 0
    for dR in dRrange:
        mesh_gm = prep_mesh(mesh_file_gm.format(dR, h1))
        mesh_meas = prep_mesh(mesh_file_gm.format(dR, H))

        R = Rref - dR

        tic()
        W, bnds, n, h, ds, _ = getFunctionSpaces(mesh_gm, elem)
        Wmeas, _, _, _, ds_meas, _ = getFunctionSpaces(mesh_meas, elem)
        print('building function spaces:  %fs' % toc())

        # interpolate uref(P2) to umeas(P1) on mesh_gm
        Vmeas = VectorFunctionSpace(Wmeas.mesh(), 'CG', 1)
        umeas = interpolate(uref, Vmeas)
        pmeas = interpolate(pref, Wmeas.sub(1).collapse())

        unorm_meas = norm(umeas.vector(), 'l2')
        pnorm_meas = norm(pmeas.vector(), 'l2')
        unorm_meas_inf = norm(umeas.vector(), 'linf')
        pnorm_meas_inf = norm(pmeas.vector(), 'linf')
        dPmeas = assemble(pmeas*ds_meas(1) - pmeas*ds_meas(2))

        tic()
        A, b, W = getLS(W, bnds, n, h, ds,
                        wall='noslip',
                        uwall=0.0,
                        # uwall=uwall(umax, R2, R1),  # noslip only
                        bctype='nitsche',
                        beta=beta,              # Nitsche only
                        elements=elem,
                        symmetric=True,
                        inlet='parabola',  # parabola, pressure
                        pin=None,            # inlet pressure
                        Rref=R,        # true reference radius inlet
                        umax=umaxbase,      # inlet velocity
                        mu=mu
                        )
        u, p = solveLS(A, b, W, 'mumps', plots=False)
        t_solv = toc()

        # compute error w.r.t. reference solution uref, pref

        # interpolate to P1 space
        # TODO LagrangeInterpolator here ?
        LI = LagrangeInterpolator()
        u1 = Function(umeas.function_space())
        p1 = Function(pmeas.function_space())
        LI.interpolate(u1, u)
        LI.interpolate(p1, p)

        # u1 = interpolate(u, umeas.function_space())
        # p1 = interpolate(p, pmeas.function_space())

        # Estimate optimal umax from (uref, u)
        # not pressure, but works for inflow, too!
        aopt = optimal_coefficient(umeas, u1)
        u1.vector()[:] *= aopt
        p1.vector()[:] *= aopt
        umaxopt.append(umaxbase*aopt)

        e_u_inf.append(norm(u1.vector() - umeas.vector(), 'linf') /
                       unorm_meas_inf)
        e_p_inf.append(norm(p1.vector() - pmeas.vector(), 'linf') /
                       pnorm_meas_inf)
        e_u_l2.append(norm(u1.vector() - umeas.vector(), 'l2')/unorm_meas)
        e_p_l2.append(norm(p1.vector() - pmeas.vector(), 'l2')/pnorm_meas)
        dP = assemble(p1*ds_meas(1) - p1*ds_meas(2))
        e_dP.append(abs(dP - dPref)/dPref)
        e_dPmeas.append(abs(dP - dPmeas)/dPmeas)

        if MPI.COMM_WORLD.rank == 0:
            print('run # %i/%i \t d = %g \t umax = %g' % (k+1, N, dR,
                                                          umaxopt[-1]))
            print('assembly and solver time:  %.4fs' % t_solv)
            print('e(dP) ref:\t %f' % e_dP[-1])
            print('e(dP) meas:\t %f' % e_dPmeas[-1])
            print('e(u)_inf:\t %f' % e_u_inf[-1])
            print('e(u)_l2: \t %f' % e_u_l2[-1])
            print('e(p)_inf:\t %f' % e_p_inf[-1])
            print('e(p)_l2: \t %f' % e_p_l2[-1])

        k += 1
        if on_cluster():
            os.remove(mesh_gm)

    if on_cluster():
        # delete mesh from /dev/shm when done
        os.remove(mesh_ref)

    shape = (len(dRrange), 1)

    return (dRrange, umaxopt,
            np.reshape(np.array(e_u_inf), shape),
            np.reshape(np.array(e_u_l2), shape),
            np.reshape(np.array(e_p_inf), shape),
            np.reshape(np.array(e_p_inf), shape),
            np.reshape(np.array(e_dP), shape),
            np.reshape(np.array(e_dPmeas), shape)), (uref, u)


def aorta_narrowed_est():
    ''' simulate over parameter grid [gamma1, gamma2, Pin]
        reference solution: big mesh, no-slip, umax = 2.0
        parameters: gamma1*a, a = (0..2)
                    gamma2*a, a = (0..2)
                    pin*a, a = (0..2), pin = 0.45
    '''
    from itertools import product

    # geometry/setup
    set_log_level(ERROR)
    mesh_reflvl = 0
    mesh_file_ref = 'coarc2d_f0.6_ref_r{0:d}.h5'.format(mesh_reflvl)
    mesh_file_gm = 'coarc2d_f0.6_d{0:g}_r{1:d}.h5'  # % (dR, mesh_reflvl)
    mesh_ref = prep_mesh(mesh_file_ref)

    # mesh_gm = prep_mesh(mesh_file_gm)

    elem = 'Mini'
    mu = 0.01
    R1 = 1.0
    R2 = 1.1
    beta = 20.

    # define parameter grid
    dRrange = [0.1, 0.15, 0.2]
    xi = 1.0
    L0 = 2.0
    gamma_split = Expression('x[0] < xi || x[0] > xi+L ? \
                             A*2.0*mu*R1/(R1*R1 - R*R) :\
                             B*2.0*mu*R1/(R1*R1 - R*R)',
                             mu=mu, R1=R1, R=R2, xi=xi, L=L0, A=1.0, B=1.0)
    pbase = 0.45
    g1range = np.arange(0.1, 2.0, 0.1)
    g2range = np.arange(0.1, 2.0, 0.1)
    pinrange = pbase*np.arange(0.5, 1.5, 0.1)

    g1range = np.arange(1.0, 1.5, 0.5)
    g2range = np.arange(1.0, 1.5, 0.5)
    pinrange = pbase*np.arange(0.5, 1.5, 0.5)

    # Calculate reference solution
    # umax = 2.
    W, bnds, n, h, ds, _ = getFunctionSpaces(mesh_ref, elem)

    A, b, W = getLS(W, bnds, n, h, ds,
                    wall='noslip',
                    uwall=0.0,
                    bctype='nitsche',
                    beta=beta,              # Nitsche only
                    gamma=None,            # navier-slip only
                    elements=elem,
                    symmetric=True,
                    inlet='pressure',  # parabola, const
                    Rref=R2,        # true reference radius inlet
                    umax=None,      # inlet velocity
                    pin=0.46,
                    mu=mu
                    )
    uref, pref = solveLS(A, b, W, 'mumps', plots=False)

    if elem == 'Mini':
        # interpolate solution from enriched elements to standard elements
        V = VectorFunctionSpace(W.mesh(), "CG", 1)
        uref = interpolate(uref, V)
        del(V)

    dPref = assemble(pref*ds(1) - pref*ds(2))
    print('Pressure jump dP_ref: %f' % dPref)

    # Loop over parameter grid

    e_u_inf = []
    e_u_l2 = []
    e_p_inf = []
    e_p_l2 = []
    e_dP = []
    N = len(g1range)*len(g2range)*len(pinrange)*len(dRrange)
    k = 0

    for dR in dRrange:
        i = 0
        mesh_gm = prep_mesh(mesh_file_gm.format(dR, mesh_reflvl))

        tic()
        W, bnds, n, h, ds, _ = getFunctionSpaces(mesh_gm, elem)
        print('building function spaces:  %fs' % toc())

        for g1, g2, pin in product(g1range, g2range, pinrange):

            gamma_split.A = g1
            gamma_split.B = g2
            tic()
            A, b, W = getLS(W, bnds, n, h, ds,
                            wall='navierslip',
                            gamma=gamma_split,         # navier-slip only
                            uwall=None,
                            # uwall=uwall(umax, R2, R1),  # noslip only
                            bctype='nitsche',
                            beta=beta,              # Nitsche only
                            elements=elem,
                            symmetric=True,
                            inlet='pressure',  # parabola, pressure
                            pin=pin,            # inlet pressure
                            Rref=None,        # true reference radius inlet
                            umax=None,      # inlet velocity
                            mu=mu
                            )
            u, p = solveLS(A, b, W, 'mumps', plots=False)
            t_solv = toc()

            # compute error w.r.t. reference solution uref, pref

            # interpolate/project uref,pref to W on first iteration
            if i == 0:
                if elem == 'Mini':
                    # special treatment for enriched elements
                    # https://bitbucket.org/fenics-project/dolfin/issues/489/wrong-interpolation-for-enriched-element#comment-16480392
                    ''' This effectively disables interpolation to enriched
                    space, projection from enriched to another mesh and using
                    expressions and constants in Dirichlet BCs on enriched
                    space. The rest (i.e.  projection to enriched,
                    interpolation from enriched to non-enriched on the same
                    mesh and Functions in Dirichlet values on enriched space)
                    should be working correctly.
                    '''
                    parameters['krylov_solver']['monitor_convergence'] = False
                    V = W.sub(0).collapse()
                    urefi = project(uref, V)
                else:
                    urefi = interpolate(uref, W.sub(0).collapse())
                prefi = interpolate(pref, W.sub(1).collapse())

                unorm_ref = norm(urefi.vector(), 'l2')
                pnorm_ref = norm(prefi.vector(), 'l2')
                unorm_ref_inf = norm(urefi.vector(), 'linf')
                pnorm_ref_inf = norm(prefi.vector(), 'linf')

            # du = Function(V)
            # du.assign(interpolate(u, V) - uref)
            # print(norm(du, 'l2')/norm(uref, 'l2'))
            # print(norm(u.vector() - uref.vector(), 'l2') /
            #       norm(uref.vector(), 'l2'))
            e_u_inf.append(norm(u.vector() - urefi.vector(), 'linf') /
                           unorm_ref_inf)
            e_p_inf.append(norm(p.vector() - prefi.vector(), 'linf') /
                           pnorm_ref_inf)
            e_u_l2.append(norm(u.vector() - urefi.vector(), 'l2')/unorm_ref)
            e_p_l2.append(norm(p.vector() - prefi.vector(), 'l2')/pnorm_ref)
            dP = assemble(p*ds(1) - p*ds(2))
            e_dP.append(abs(dP - dPref)/dPref)

            if MPI.COMM_WORLD.rank == 0:
                #     # ('Expression' if 'expression' in str(type(gamma)) else
                #     #  str(gamma)), gi)
                print('\nrun # %i/%i \t d = %g' % ((i+1)*(k+1), N, dR))
                print('assembly and solver time:  %.4fs' % t_solv)
                print('e(dP):   \t %f' % e_dP[-1])
                print('e(u)_inf:\t %f' % e_u_inf[-1])
                print('e(u)_l2: \t %f' % e_u_l2[-1])
                print('e(p)_inf:\t %f' % e_p_inf[-1])
                print('e(p)_l2: \t %f' % e_p_l2[-1])

            i += 1

        k += 1

        if on_cluster():
            # delete mesh from /dev/shm when done
            os.remove(mesh_gm)

    if on_cluster():
        # delete mesh from /dev/shm when done
        os.remove(mesh_ref)

    n4 = len(dRrange)
    n1 = len(g1range)
    n2 = len(g2range)
    n3 = len(pinrange)
    shape = (n4, n1, n2, n3)

    return (g1range, g2range, pinrange, dRrange,
            np.reshape(np.array(e_u_inf), shape),
            np.reshape(np.array(e_u_l2), shape),
            np.reshape(np.array(e_p_inf), shape),
            np.reshape(np.array(e_p_inf), shape),
            np.reshape(np.array(e_dP), shape)), (uref, u)


def aorta_narrowed_est_popt(recalc=False, mesh_reflvl=0):
    ''' Simulate over parameter grid [gamma1, gamma2]
        Reference solution: big mesh, no-slip, p_in
        parameters: gamma1*a, a = (0..2)
                    gamma2*a, a = (0..2)

        For each set (gamma1, gamma2), calculate (u,p) with guessed p_in (e.g.,
        p_in = 1). Then, using optimal_coefficient(), calculate p* = a*p_in.
        The solution u1 = a*u should coincide with the solution of the Stokes
        problem with inlet pressure p_in' = a*p_in.
    '''
    from itertools import product

    # geometry/setup
    set_log_level(ERROR)
    mesh_file_ref = 'coarc2d_f0.6_ref_r{0:d}.h5'.format(mesh_reflvl)
    mesh_file_gm = 'coarc2d_f0.6_d{0:g}_r{1:d}.h5'  # .format(dR, mesh_reflvl)
    mesh_ref = prep_mesh(mesh_file_ref)

    # mesh_gm = prep_mesh(mesh_file_gm)

    elem = 'Mini'
    mu = 0.01
    R1 = 1.0
    R2 = 1.1
    beta = 20.

    # define parameter grid
    dRrange = [0.1, 0.15]  # , 0.15, 0.2]
    xi = 1.0
    L0 = 2.0
    gamma_split = Expression('x[0] < xi || x[0] > xi+L ? \
                             A*2.0*mu*R1/(R1*R1 - R*R) :\
                             B*2.0*mu*R1/(R1*R1 - R*R)',
                             mu=mu, R1=R1, R=R2, xi=xi, L=L0, A=1.0, B=1.0)
    pbase = 1.0
    g1range = np.arange(0.1, 2.1, 0.1)
    g2range = np.arange(0.1, 2.1, 0.1)

    # g1range = [1.0]
    # g2range = [0.4]
    # g1range = np.arange(1.0, 1.5, 0.5)
    # g2range = np.arange(1.0, 1.5, 0.5)

    # TODO: CALCULATE OPTIMAL PRESSURE automatically for EACH MESH!

    # Calculate reference solution
    # umax = 2.
    W, bnds, n, h, ds, _ = getFunctionSpaces(mesh_ref, elem)

    A, b, W = getLS(W, bnds, n, h, ds,
                    wall='noslip',
                    uwall=0.0,
                    bctype='nitsche',
                    beta=beta,              # Nitsche only
                    gamma=None,            # navier-slip only
                    elements=elem,
                    symmetric=True,
                    inlet='pressure',  # parabola, const
                    Rref=R2,        # true reference radius inlet
                    umax=None,      # inlet velocity
                    pin=0.46,
                    mu=mu
                    )
    uref, pref = solveLS(A, b, W, 'mumps', plots=False)

    if elem == 'Mini':
        # interpolate solution from enriched elements to standard elements
        V = VectorFunctionSpace(W.mesh(), "CG", 1)
        uref = interpolate(uref, V)
        del(V)

    dPref = assemble(pref*ds(1) - pref*ds(2))
    print('Pressure jump dP_ref: %f' % dPref)

    # Loop over parameter grid

    e_u_inf = []
    e_u_l2 = []
    e_p_inf = []
    e_p_l2 = []
    e_dP = []
    pin = []
    N = len(g1range)*len(g2range)*len(dRrange)
    k = 0

    for dR in dRrange:
        i = 0
        mesh_gm = prep_mesh(mesh_file_gm.format(dR, mesh_reflvl))

        tic()
        W, bnds, n, h, ds, _ = getFunctionSpaces(mesh_gm, elem)
        print('building function spaces:  %fs' % toc())

        for g1, g2 in product(g1range, g2range):

            gamma_split.A = g1
            gamma_split.B = g2
            tic()
            # compute (u, p)(g1, g2) with p_in = pbase = 1 (e.g.)
            A, b, W = getLS(W, bnds, n, h, ds,
                            wall='navierslip',
                            gamma=gamma_split,         # navier-slip only
                            uwall=None,
                            # uwall=uwall(umax, R2, R1),  # noslip only
                            bctype='nitsche',
                            beta=beta,              # Nitsche only
                            elements=elem,
                            symmetric=True,
                            inlet='pressure',  # parabola, pressure
                            pin=pbase,            # inlet pressure
                            Rref=None,        # true reference radius inlet
                            umax=None,      # inlet velocity
                            mu=mu
                            )
            u, p = solveLS(A, b, W, 'mumps', plots=False)
            t_solv = toc()

            # compute error w.r.t. reference solution uref, pref

            # interpolate/project uref,pref to W on first iteration
            if i == 0:
                if elem == 'Mini':
                    # special treatment for enriched elements
                    # https://bitbucket.org/fenics-project/dolfin/issues/489/wrong-interpolation-for-enriched-element#comment-16480392
                    ''' This effectively disables interpolation to enriched
                    space, projection from enriched to another mesh and using
                    expressions and constants in Dirichlet BCs on enriched
                    space. The rest (i.e.  projection to enriched,
                    interpolation from enriched to non-enriched on the same
                    mesh and Functions in Dirichlet values on enriched space)
                    should be working correctly.
                    '''
                    parameters['krylov_solver']['monitor_convergence'] = False
                    # V = W.sub(0).collapse()
                    # urefi = project(uref, V)
                    urefi = project(uref, u.function_space())
                else:
                    # urefi = interpolate(uref, W.sub(0).collapse())
                    urefi = interpolate(uref, u.function_space())
                prefi = interpolate(pref, p.function_space())

                unorm_ref = norm(urefi.vector(), 'l2')
                pnorm_ref = norm(prefi.vector(), 'l2')
                unorm_ref_inf = norm(urefi.vector(), 'linf')
                pnorm_ref_inf = norm(prefi.vector(), 'linf')

            # Estimate optimal pressure from (uref, u)
            a_opt = optimal_coefficient(urefi, u)

            # now, either recalculate (u,p) with corrected p_in or just
            # multiply (u, p) by a_opt

            if recalc:
                A, b, W = getLS(W, bnds, n, h, ds,
                                wall='navierslip',
                                gamma=gamma_split,         # navier-slip only
                                uwall=None,
                                # uwall=uwall(umax, R2, R1),  # noslip only
                                bctype='nitsche',
                                beta=beta,              # Nitsche only
                                elements=elem,
                                symmetric=True,
                                inlet='pressure',  # parabola, pressure
                                pin=pbase*a_opt,            # inlet pressure
                                Rref=None,        # true reference radius inlet
                                umax=None,      # inlet velocity
                                mu=mu
                                )
                u, p = solveLS(A, b, W, 'mumps', plots=False)
            else:
                u.vector()[:] *= a_opt
                p.vector()[:] *= a_opt

            pin.append(a_opt*pbase)

            # du = Function(V)
            # du.assign(interpolate(u, V) - uref)
            # print(norm(du, 'l2')/norm(uref, 'l2'))
            # print(norm(u.vector() - uref.vector(), 'l2') /
            #       norm(uref.vector(), 'l2'))
            e_u_inf.append(norm(u.vector() - urefi.vector(), 'linf') /
                           unorm_ref_inf)
            e_p_inf.append(norm(p.vector() - prefi.vector(), 'linf') /
                           pnorm_ref_inf)
            e_u_l2.append(norm(u.vector() - urefi.vector(), 'l2')/unorm_ref)
            e_p_l2.append(norm(p.vector() - prefi.vector(), 'l2')/pnorm_ref)
            dP = assemble(p*ds(1) - p*ds(2))
            e_dP.append(abs(dP - dPref)/dPref)

            if MPI.COMM_WORLD.rank == 0:
                #     # ('Expression' if 'expression' in str(type(gamma)) else
                #     #  str(gamma)), gi)
                print('run # %i/%i \t d = %g \t (g1, g2) = (%.2g, %.2g)' %
                      (i+1+k*len(g1range)*len(g2range), N, dR, g1, g2))
                print('assembly and solver time:  %.4fs' % t_solv)
                print('e(dP):   \t %f' % e_dP[-1])
                print('e(u)_inf:\t %f' % e_u_inf[-1])
                print('e(u)_l2: \t %f' % e_u_l2[-1])
                print('e(p)_inf:\t %f' % e_p_inf[-1])
                print('e(p)_l2: \t %f\n' % e_p_l2[-1])

            i += 1

        k += 1

        if on_cluster():
            # delete mesh from /dev/shm when done
            os.remove(mesh_gm)

    if on_cluster():
        # delete mesh from /dev/shm when done
        os.remove(mesh_ref)

    n1 = len(dRrange)
    n2 = len(g1range)
    n3 = len(g2range)
    shape = (n1, n2, n3)

    return (g1range, g2range, dRrange, np.reshape(np.array(pin), shape),
            np.reshape(np.array(e_u_inf), shape),
            np.reshape(np.array(e_u_l2), shape),
            np.reshape(np.array(e_p_inf), shape),
            np.reshape(np.array(e_p_inf), shape),
            np.reshape(np.array(e_dP), shape)), (uref, u)


def aorta_narrowed_est_inflow(mesh_reflvl=0):
    ''' Brute force method.
        simulate over parameter grid [gamma1, gamma2, Pin]
        reference solution: big mesh, no-slip, umax = 2.0
        parameters: gamma1*a, a = (0..2)
                    gamma2*a, a = (0..2)
                    pin*a, a = (0..2), pin = 0.45
    '''
    from itertools import product

    # geometry/setup
    set_log_level(ERROR)
    mesh_reflvl = 0
    mesh_file_ref = 'coarc2d_f0.6_ref_r{0:d}.h5'.format(mesh_reflvl)
    mesh_file_gm = 'coarc2d_f0.6_d{0:g}_r{1:d}.h5'  # % (dR, mesh_reflvl)
    mesh_ref = prep_mesh(mesh_file_ref)

    # mesh_gm = prep_mesh(mesh_file_gm)

    elem = 'Mini'
    mu = 0.01
    Rref = 1.0
    # R2 = 1.1
    beta = 20.

    # define parameter grid
    dRrange = [0.1]  # , 0.15, 0.2]
    xi = 1.0
    L0 = 2.0

    # two options, based on geometric poiseuille gamma:
    # 1. 'static' gamma: f.a. other params, iterate over same g1,g2 grid
    umaxbase = 2.0
    Rinbase = 1.0
    R = Rref - np.array(dRrange).mean()
    gmbase = 2.0*mu*R/(R**2 - Rref**2)
    # gamma_split = Expression('x[0] < xi || x[0] > xi+L ? \
    #                          A*2.0*mu*R1/(R1*R1 - R*R) :\
    #                          B*2.0*mu*R1/(R1*R1 - R*R)', mu=mu,
    #                          R1=Rref, R=R2, xi=xi, L=L0, A=1.0, B=1.0)
    gamma_split = Expression('x[0] < xi || x[0] > xi+L ? A*Gm : B*Gm',
                             mu=mu, xi=xi, L=L0, Gm=gmbase, A=1.0, B=1.0)
    g1range = np.arange(0.1, 2.0, 0.25)
    g2range = np.arange(0.1, 2.0, 0.25)
    # 2. set R1, R according to current Rin, dR iteration

    umaxrange = umaxbase*np.arange(0.5, 1.5, 0.25)
    Rinrange = Rinbase*np.arange(0.5, 1.5, 0.25)

    # Calculate reference solution
    W, bnds, n, h, ds, _ = getFunctionSpaces(mesh_ref, elem)

    A, b, W = getLS(W, bnds, n, h, ds,
                    wall='noslip',
                    uwall=0.0,
                    bctype='nitsche',
                    beta=beta,              # Nitsche only
                    gamma=None,            # navier-slip only
                    elements=elem,
                    symmetric=True,
                    inlet='parabola',  # parabola, const
                    Rref=1.0,        # true reference radius inlet
                    umax=2.0,      # inlet velocity
                    pin=None,
                    mu=mu
                    )
    uref, pref = solveLS(A, b, W, 'mumps', plots=False)

    if elem == 'Mini':
        # interpolate solution from enriched elements to standard elements
        V = VectorFunctionSpace(W.mesh(), "CG", 1)
        uref = interpolate(uref, V)
        del(V)

    dPref = assemble(pref*ds(1) - pref*ds(2))
    print('Pressure jump dP_ref: %f' % dPref)

    # Loop over parameter grid

    e_u_inf = []
    e_u_l2 = []
    e_p_inf = []
    e_p_l2 = []
    e_dP = []
    N = len(g1range)*len(g2range)*len(Rinrange)*len(dRrange)*len(umaxrange)
    Np = len(g1range)*len(g2range)*len(Rinrange)*len(umaxrange)
    k = 0

    for dR in dRrange:
        i = 0
        mesh_gm = prep_mesh(mesh_file_gm.format(dR, mesh_reflvl))

        tic()
        W, bnds, n, h, ds, _ = getFunctionSpaces(mesh_gm, elem)
        print('building function spaces:  %fs' % toc())

        for g1, g2, umax, Rin in product(g1range, g2range, umaxrange,
                                         Rinrange):

            gamma_split.A = g1
            gamma_split.B = g2
            tic()
            A, b, W = getLS(W, bnds, n, h, ds,
                            wall='navierslip',
                            gamma=gamma_split,         # navier-slip only
                            uwall=None,
                            bctype='nitsche',
                            beta=beta,              # Nitsche only
                            elements=elem,
                            symmetric=True,
                            inlet='parabola',  # parabola, pressure
                            Rref=Rin,        # true reference radius inlet
                            umax=umax,      # inlet velocity
                            pin=None,            # inlet pressure
                            mu=mu
                            )
            u, p = solveLS(A, b, W, 'mumps', plots=False)
            t_solv = toc()

            # compute error w.r.t. reference solution uref, pref

            # interpolate/project uref,pref to W on first iteration
            if i == 0:
                if elem == 'Mini':
                    # special treatment for enriched elements
                    # https://bitbucket.org/fenics-project/dolfin/issues/489/wrong-interpolation-for-enriched-element#comment-16480392
                    ''' This effectively disables interpolation to enriched
                    space, projection from enriched to another mesh and using
                    expressions and constants in Dirichlet BCs on enriched
                    space. The rest (i.e.  projection to enriched,
                    interpolation from enriched to non-enriched on the same
                    mesh and Functions in Dirichlet values on enriched space)
                    should be working correctly.
                    '''
                    parameters['krylov_solver']['monitor_convergence'] = False
                    V = W.sub(0).collapse()
                    urefi = project(uref, V)
                else:
                    urefi = interpolate(uref, W.sub(0).collapse())
                prefi = interpolate(pref, W.sub(1).collapse())

                unorm_ref = norm(urefi.vector(), 'l2')
                pnorm_ref = norm(prefi.vector(), 'l2')
                unorm_ref_inf = norm(urefi.vector(), 'linf')
                pnorm_ref_inf = norm(prefi.vector(), 'linf')

            # du = Function(V)
            # du.assign(interpolate(u, V) - uref)
            # print(norm(du, 'l2')/norm(uref, 'l2'))
            # print(norm(u.vector() - uref.vector(), 'l2') /
            #       norm(uref.vector(), 'l2'))
            e_u_inf.append(norm(u.vector() - urefi.vector(), 'linf') /
                           unorm_ref_inf)
            e_p_inf.append(norm(p.vector() - prefi.vector(), 'linf') /
                           pnorm_ref_inf)
            e_u_l2.append(norm(u.vector() - urefi.vector(), 'l2')/unorm_ref)
            e_p_l2.append(norm(p.vector() - prefi.vector(), 'l2')/pnorm_ref)
            dP = assemble(p*ds(1) - p*ds(2))
            e_dP.append(abs(dP - dPref)/dPref)

            if MPI.COMM_WORLD.rank == 0:
                #     # ('Expression' if 'expression' in str(type(gamma)) else
                #     #  str(gamma)), gi)
                print('\nrun # %i/%i \t d = %g,\t umax = %g,\t Rin = %g' %
                      (i+1 + k*Np, N, dR, umax, Rin))
                print('assembly and solver time:  %.4fs' % t_solv)
                print('e(dP):   \t %f' % e_dP[-1])
                print('e(u)_inf:\t %f' % e_u_inf[-1])
                print('e(u)_l2: \t %f' % e_u_l2[-1])
                print('e(p)_inf:\t %f' % e_p_inf[-1])
                print('e(p)_l2: \t %f' % e_p_l2[-1])

            i += 1
        k += 1

        if on_cluster():
            # delete mesh from /dev/shm when done
            os.remove(mesh_gm)

    if on_cluster():
        # delete mesh from /dev/shm when done
        os.remove(mesh_ref)

    n1 = len(dRrange)
    n2 = len(g1range)
    n3 = len(g2range)
    n4 = len(umaxrange)
    n5 = len(Rinrange)
    shape = (n1, n2, n3, n4, n5)

    return (g1range, g2range, umaxrange, Rinrange, dRrange,
            np.reshape(np.array(e_u_inf), shape),
            np.reshape(np.array(e_u_l2), shape),
            np.reshape(np.array(e_p_inf), shape),
            np.reshape(np.array(e_p_inf), shape),
            np.reshape(np.array(e_dP), shape)), (uref, u)


def aorta_narrowed_est_inflow_uopt(mesh_reflvl=0):
    ''' Parabolic inflow profile (umax, Rin).
        Simulate over parameter grid [dR(mesh), gamma1, gamma2, Rin]
        Calculate optimal umax from a = optimal_coefficient()
            -> (u*, p*) = a*(u, p)
        reference solution: big mesh R=1, no-slip, umax = 2.0
    '''
    from itertools import product

    # geometry/setup
    set_log_level(ERROR)
    mesh_reflvl = 0
    mesh_file_ref = 'coarc2d_f0.6_ref_r{0:d}.h5'.format(mesh_reflvl)
    mesh_file_gm = 'coarc2d_f0.6_d{0:g}_r{1:d}.h5'  # % (dR, mesh_reflvl)
    mesh_ref = prep_mesh(mesh_file_ref)

    # mesh_gm = prep_mesh(mesh_file_gm)

    elem = 'Mini'
    mu = 0.01
    Rref = 1.0
    # R2 = 1.1
    beta = 20.

    # define parameter grid
    dRrange = [0.1, 0.15, 0.2]
    xi = 1.0
    L0 = 2.0

    # two options, based on geometric poiseuille gamma:
    # 1. 'static' gamma: f.a. other params, iterate over same g1,g2 grid
    umax = 1.3
    # Rinbase = 1.0
    R = Rref - np.array(dRrange).mean()
    gmbase = 2.0*mu*R/(R**2 - Rref**2)
    # gamma_split = Expression('x[0] < xi || x[0] > xi+L ? \
    #                          A*2.0*mu*R1/(R1*R1 - R*R) :\
    #                          B*2.0*mu*R1/(R1*R1 - R*R)', mu=mu,
    #                          R1=Rref, R=R2, xi=xi, L=L0, A=1.0, B=1.0)
    gamma_split = Expression('x[0] < xi || x[0] > xi+L ? A*Gm : B*Gm',
                             mu=mu, xi=xi, L=L0, Gm=gmbase, A=1.0, B=1.0)
    g1range = np.arange(0.1, 2.0, 0.1)
    g2range = np.arange(0.1, 2.0, 0.1)
    # 2. set R1, R according to current Rin, dR iteration

    # WATCH OUT! Rin < R LEADS TO NEGATIVE VELOCITIES!
    # Vary from smallest Radius min(R) Rref - max(dR) to max(R)*factor or Rref!
    # TODO THIS
    Rmin = Rref - max(dRrange)
    Rmax = Rref - min(dRrange)
    Rinrange = np.linspace(Rmin, Rmax*1.2, 20)
    # Rinrange = np.linspace(Rmin, Rref, 10)
    # Rinrange = Rinbase*np.arange(1.0, 1.5, 0.05)

    # Calculate reference solution
    W, bnds, n, h, ds, _ = getFunctionSpaces(mesh_ref, elem)

    A, b, W = getLS(W, bnds, n, h, ds,
                    wall='noslip',
                    uwall=0.0,
                    bctype='nitsche',
                    beta=beta,              # Nitsche only
                    gamma=None,            # navier-slip only
                    elements=elem,
                    symmetric=True,
                    inlet='parabola',  # parabola, const
                    Rref=1.0,        # true reference radius inlet
                    umax=2.0,      # inlet velocity
                    pin=None,
                    mu=mu
                    )
    uref, pref = solveLS(A, b, W, 'mumps', plots=False)

    if elem == 'Mini':
        # interpolate solution from enriched elements to standard elements
        V = VectorFunctionSpace(W.mesh(), "CG", 1)
        uref = interpolate(uref, V)
        del(V)

    dPref = assemble(pref*ds(1) - pref*ds(2))
    print('Pressure jump dP_ref: %f' % dPref)

    # Loop over parameter grid

    e_u_inf = []
    e_u_l2 = []
    e_p_inf = []
    e_p_l2 = []
    e_dP = []
    umaxopt = []
    N = len(g1range)*len(g2range)*len(Rinrange)*len(dRrange)
    Np = len(g1range)*len(g2range)*len(Rinrange)
    k = 0

    for dR in dRrange:
        i = 0
        mesh_gm = prep_mesh(mesh_file_gm.format(dR, mesh_reflvl))

        tic()
        W, bnds, n, h, ds, _ = getFunctionSpaces(mesh_gm, elem)
        print('building function spaces:  %fs' % toc())

        for g1, g2, Rin in product(g1range, g2range, Rinrange):

            gamma_split.A = g1
            gamma_split.B = g2
            tic()
            A, b, W = getLS(W, bnds, n, h, ds,
                            wall='navierslip',
                            gamma=gamma_split,         # navier-slip only
                            uwall=None,
                            bctype='nitsche',
                            beta=beta,              # Nitsche only
                            elements=elem,
                            symmetric=True,
                            inlet='parabola',  # parabola, pressure
                            Rref=Rin,        # true reference radius inlet
                            umax=umax,      # inlet velocity
                            pin=None,            # inlet pressure
                            mu=mu
                            )
            u, p = solveLS(A, b, W, 'mumps', plots=False)
            t_solv = toc()

            # compute error w.r.t. reference solution uref, pref

            # interpolate/project uref,pref to W on first iteration
            if i == 0:
                if elem == 'Mini':
                    # special treatment for enriched elements
                    # https://bitbucket.org/fenics-project/dolfin/issues/489/wrong-interpolation-for-enriched-element#comment-16480392
                    ''' This effectively disables interpolation to enriched
                    space, projection from enriched to another mesh and using
                    expressions and constants in Dirichlet BCs on enriched
                    space. The rest (i.e.  projection to enriched,
                    interpolation from enriched to non-enriched on the same
                    mesh and Functions in Dirichlet values on enriched space)
                    should be working correctly.
                    '''
                    parameters['krylov_solver']['monitor_convergence'] = False
                    V = W.sub(0).collapse()
                    urefi = project(uref, V)
                else:
                    urefi = interpolate(uref, W.sub(0).collapse())
                prefi = interpolate(pref, W.sub(1).collapse())

                unorm_ref = norm(urefi.vector(), 'l2')
                pnorm_ref = norm(prefi.vector(), 'l2')
                unorm_ref_inf = norm(urefi.vector(), 'linf')
                pnorm_ref_inf = norm(prefi.vector(), 'linf')

            aopt = optimal_coefficient(urefi, u)
            u.vector()[:] *= aopt
            p.vector()[:] *= aopt
            umaxopt.append(umax*aopt)

            # du = Function(V)
            # du.assign(interpolate(u, V) - uref)
            # print(norm(du, 'l2')/norm(uref, 'l2'))
            # print(norm(u.vector() - uref.vector(), 'l2') /
            #       norm(uref.vector(), 'l2'))
            e_u_inf.append(norm(u.vector() - urefi.vector(), 'linf') /
                           unorm_ref_inf)
            e_p_inf.append(norm(p.vector() - prefi.vector(), 'linf') /
                           pnorm_ref_inf)
            e_u_l2.append(norm(u.vector() - urefi.vector(), 'l2')/unorm_ref)
            e_p_l2.append(norm(p.vector() - prefi.vector(), 'l2')/pnorm_ref)
            dP = assemble(p*ds(1) - p*ds(2))
            e_dP.append(abs(dP - dPref)/dPref)

            if MPI.COMM_WORLD.rank == 0:
                #     # ('Expression' if 'expression' in str(type(gamma)) else
                #     #  str(gamma)), gi)
                print('\nrun # %i/%i \t d = %g,\t umax = %g,\t Rin = %g' %
                      (i+1 + k*Np, N, dR, umax*aopt, Rin))
                print('assembly and solver time:  %.4fs' % t_solv)
                print('e(dP):   \t %f' % e_dP[-1])
                print('e(u)_inf:\t %f' % e_u_inf[-1])
                print('e(u)_l2: \t %f' % e_u_l2[-1])
                print('e(p)_inf:\t %f' % e_p_inf[-1])
                print('e(p)_l2: \t %f' % e_p_l2[-1])

            i += 1
        k += 1

        if on_cluster():
            # delete mesh from /dev/shm when done
            os.remove(mesh_gm)

    if on_cluster():
        # delete mesh from /dev/shm when done
        os.remove(mesh_ref)

    n1 = len(dRrange)
    n2 = len(g1range)
    n3 = len(g2range)
    n4 = len(Rinrange)
    shape = (n1, n2, n3, n4)

    return (g1range, g2range, Rinrange, dRrange,
            np.reshape(np.array(umaxopt), shape),
            np.reshape(np.array(e_u_inf), shape),
            np.reshape(np.array(e_u_l2), shape),
            np.reshape(np.array(e_p_inf), shape),
            np.reshape(np.array(e_p_inf), shape),
            np.reshape(np.array(e_dP), shape)), (uref, u)


def aorta_narrowed_est_inflow_uopt_measrmt(h1=0.1):
    ''' Emulate real measurement.
            uref: P2(Omega^h) -> R^d
            umeas: P1(Omega_i^H) -> R^d, umeas = I1^H uref  "real measurement"
            e.g. h = 0.05, H = 0.15

            STEint: P1 Bubble(Omega_i^H)
            u_ns, u_gm: P1 Bubble(Omega_i^h2), e.g. h2 = (h+H)/2

    Parabolic inflow profile (umax, Rin).
        Simulate over parameter grid [dR(mesh), gamma1, gamma2, Rin]
        Calculate optimal umax from a = optimal_coefficient()
            -> (u*, p*) = a*(u, p)
        reference solution: big mesh R=1, no-slip, umax = 2.0
    '''
    from itertools import product

    # geometry/setup
    set_log_level(ERROR)
    mesh_file_ref = 'coarc2d_f0.6_ref_r{0:d}.h5'.format(0)
    # mesh_file_meas = 'coarc2d_f0.6_d{0:g}_h{1:g}.h5'
    mesh_file_gm = 'coarc2d_f0.6_d{0:g}_h{1:g}.h5'
    mesh_ref = prep_mesh(mesh_file_ref)

    # mesh_gm = prep_mesh(mesh_file_gm)

    H = 0.15
    elem = 'Mini'
    elem_ref = 'TH'
    mu = 0.01
    Rref = 1.0
    beta = 20.

    # define parameter grid
    dRrange = [0.1, 0.2]
    xi = 1.0
    L0 = 2.0

    # two options, based on geometric poiseuille gamma:
    # 1. 'static' gamma: f.a. other params, iterate over same g1,g2 grid
    umax = 1.3
    # Rinbase = 1.0
    R = Rref - np.array(dRrange).mean()
    gmbase = 2.0*mu*R/(R**2 - Rref**2)
    # gamma_split = Expression('x[0] < xi || x[0] > xi+L ? \
    #                          A*2.0*mu*R1/(R1*R1 - R*R) :\
    #                          B*2.0*mu*R1/(R1*R1 - R*R)', mu=mu,
    #                          R1=Rref, R=R2, xi=xi, L=L0, A=1.0, B=1.0)
    gamma_split = Expression('x[0] < xi || x[0] > xi+L ? A*Gm : B*Gm',
                             mu=mu, xi=xi, L=L0, Gm=gmbase, A=1.0, B=1.0)
    g1range = np.arange(0.4, 1.8, 0.1)
    g2range = np.arange(0.4, 1.8, 0.1)
    # g1range = np.array([0.6, 1.0, 1.7])
    # g2range = np.array([0.6, 1., 1.4])
    # 2. set R1, R according to current Rin, dR iteration

    # WATCH OUT! Rin < R LEADS TO NEGATIVE VELOCITIES!
    # Vary from smallest Radius min(R) Rref - max(dR) to max(R)*factor or Rref!
    # TODO THIS
    Rmin = Rref - max(dRrange)
    Rmax = (Rref - min(dRrange))*1.2
    Rmin = 0.95
    Rmax = 1.05
    Rinrange = np.linspace(Rmin, Rmax, 3)
    Rinrange = [1.0]

    # Calculate reference solution
    W, bnds, n, h, ds, _ = getFunctionSpaces(mesh_ref, elem_ref)

    A, b, W = getLS(W, bnds, n, h, ds,
                    wall='noslip',
                    uwall=0.0,
                    bctype='nitsche',
                    beta=beta,              # Nitsche only
                    gamma=None,            # navier-slip only
                    elements=elem_ref,
                    symmetric=True,
                    inlet='parabola',  # parabola, const
                    Rref=1.0,        # true reference radius inlet
                    umax=2.0,      # inlet velocity
                    pin=None,
                    mu=mu
                    )
    uref, pref = solveLS(A, b, W, 'mumps', plots=False)

    if elem_ref == 'Mini':
        # interpolate solution from enriched elements to standard elements
        V = VectorFunctionSpace(W.mesh(), "CG", 1)
        uref = interpolate(uref, V)
        del(V)

    dPref = assemble(pref*ds(1) - pref*ds(2))
    print('Pressure jump dP_ref: %f' % dPref)

    # Loop over parameter grid

    e_u_inf = []
    e_u_l2 = []
    e_p_inf = []
    e_p_l2 = []
    e_dP = []
    e_dPmeas = []
    umaxopt = []
    N = len(g1range)*len(g2range)*len(Rinrange)*len(dRrange)
    Np = len(g1range)*len(g2range)*len(Rinrange)
    k = 0

    for dR in dRrange:
        i = 0
        mesh_gm = prep_mesh(mesh_file_gm.format(dR, h1))
        mesh_meas = prep_mesh(mesh_file_gm.format(dR, H))

        tic()
        W, bnds, n, h, ds, _ = getFunctionSpaces(mesh_gm, elem)
        Wmeas, _, _, _, ds_meas, _ = getFunctionSpaces(mesh_meas, elem)
        print('building function spaces:  %fs' % toc())

        # interpolate uref(P2) to umeas(P1) on mesh_gm
        Vmeas = VectorFunctionSpace(Wmeas.mesh(), 'CG', 1)
        umeas = interpolate(uref, Vmeas)
        pmeas = interpolate(pref, Wmeas.sub(1).collapse())

        unorm_meas = norm(umeas.vector(), 'l2')
        pnorm_meas = norm(pmeas.vector(), 'l2')
        unorm_meas_inf = norm(umeas.vector(), 'linf')
        pnorm_meas_inf = norm(pmeas.vector(), 'linf')
        dPmeas = assemble(pmeas*ds_meas(1) - pmeas*ds_meas(2))

        for g1, g2, Rin in product(g1range, g2range, Rinrange):

            gamma_split.A = g1
            gamma_split.B = g2
            tic()
            A, b, W = getLS(W, bnds, n, h, ds,
                            wall='navierslip',
                            gamma=gamma_split,         # navier-slip only
                            uwall=None,
                            bctype='nitsche',
                            beta=beta,              # Nitsche only
                            elements=elem,
                            symmetric=True,
                            inlet='parabola',  # parabola, pressure
                            Rref=Rin,        # true reference radius inlet
                            umax=umax,      # inlet velocity
                            pin=None,            # inlet pressure
                            mu=mu
                            )
            u, p = solveLS(A, b, W, 'mumps', plots=False)
            t_solv = toc()

            # compute error w.r.t. reference solution umeas, pmeas
            # interpolate to P1 space on coarse H grid
            LI = LagrangeInterpolator()
            u1 = Function(umeas.function_space())
            p1 = Function(pmeas.function_space())
            LI.interpolate(u1, u)
            LI.interpolate(p1, p)
            # u1 = interpolate(u, umeas.function_space())
            # p1 = interpolate(p, pmeas.function_space())

            aopt = optimal_coefficient(umeas, u1)
            u1.vector()[:] *= aopt
            p1.vector()[:] *= aopt

            dP = assemble(p1*ds_meas(1) - p1*ds_meas(2))

            umaxopt.append(umax*aopt)
            e_u_inf.append(norm(u1.vector() - umeas.vector(), 'linf') /
                           unorm_meas_inf)
            e_p_inf.append(norm(p1.vector() - pmeas.vector(), 'linf') /
                           pnorm_meas_inf)
            e_u_l2.append(norm(u1.vector() - umeas.vector(), 'l2')/unorm_meas)
            e_p_l2.append(norm(p1.vector() - pmeas.vector(), 'l2')/pnorm_meas)
            e_dP.append(abs(dP - dPref)/dPref)
            e_dPmeas.append(abs(dP - dPmeas)/dPmeas)

            if MPI.COMM_WORLD.rank == 0:
                #     # ('Expression' if 'expression' in str(type(gamma)) else
                #     #  str(gamma)), gi)
                print('\nrun # %i/%i \t d = %g,\t umax = %g,\t Rin = %g' %
                      (i+1 + k*Np, N, dR, umax*aopt, Rin))
                print('assembly and solver time:  %.4fs' % t_solv)
                print('e(dP) ref:\t %f \t ***' % e_dP[-1])
                print('e(dP) meas:\t %f' % e_dPmeas[-1])
                print('e(u)_inf:\t %f' % e_u_inf[-1])
                print('e(u)_l2: \t %f \t ***' % e_u_l2[-1])
                print('e(p)_inf:\t %f' % e_p_inf[-1])
                print('e(p)_l2: \t %f' % e_p_l2[-1])

            i += 1
        k += 1

        if on_cluster():
            # delete mesh from /dev/shm when done
            os.remove(mesh_gm)

    if on_cluster():
        # delete mesh from /dev/shm when done
        os.remove(mesh_ref)

    n1 = len(dRrange)
    n2 = len(g1range)
    n3 = len(g2range)
    n4 = len(Rinrange)
    shape = (n1, n2, n3, n4)

    return (g1range, g2range, Rinrange, dRrange,
            np.reshape(np.array(umaxopt), shape),
            np.reshape(np.array(e_u_inf), shape),
            np.reshape(np.array(e_u_l2), shape),
            np.reshape(np.array(e_p_inf), shape),
            np.reshape(np.array(e_p_inf), shape),
            np.reshape(np.array(e_dP), shape),
            np.reshape(np.array(e_dPmeas), shape)), (uref, u)


def optimal_coefficient(uref, u1):
    ''' Stokes linearity: if (u1, p1) is a solution to the Stokes problem, so
        is (a*u1, a*p1), a = const \in R. Hence, for "any" pressure ph,
        solution to the Stokes problem, we have a corresponding velocity field
        uh = a*u1.
        Given a velocity field u = u_ref, we want to estimate the corresponding
        pressure field. The L2 error of velocity  ||u_ref - a*u1||_L2 can be
        minimized for the parameter a:
            minimize the functional
                J(a) = ||u_ref - a*u1||^2_L2 = (u_ref - a*u1, u_ref - a*u1)
             or J(a) = <U_ref - a*U1, M*(U_ref - a*U1)>,
                where U_x are the DOF vectors and M is the mass matrix,
                (.,.) the L2 inner product and <.,.> euclid
        We get J(a) = <U, M*U> - 2*a*<U, M*U1> + a^2*<U1, M*U1> and
            dJ/da = -2*<U, M*U1> + 2*a*<U1, M*U1> =!= 0
            <=> a_opt = <U, M*U1>/<U1, M*U1>
        a_opt is the factor of the pressure (a_opt*p1) that minimizes the L2
        error w.r.t. the reference velocity, p1 being the pressure
        corresponding to the u1 velocity field (arbitrary).

        So, workflow (g1, g2):
            1) get/have uref
            2) calculate u1 with guessed p1
            3) obtain a_opt
            4) with p_in = p1*a_opt, calculate err(u) for g1 x g2 grid
            5) determine optimal set of (g1, g2)

    '''

    Vr = uref.function_space()
    V = u1.function_space()
    if not Vr == V:
        if str(type(Vr.ufl_element())).split('.')[-1] == 'EnrichedElement':
            # interpolate to non-enriched FS on same mesh first, then project
            Vr = VectorFunctionSpace(Vr.mesh(), 'CG', 1)
            uref = interpolate(uref, Vr)
            uref = project(uref, V)
        else:
            uref = interpolate(uref, V)
    else:
        # print("don't interpolate, same function spaces")
        pass

    # needed?
    # dx = Measure('dx', domain=V.mesh())

    def l2(u, v):
        return assemble(inner(u, v)*dx)

    # go via mass matrix:
    # w = TrialFunction(V)
    # v = TestFunction(V)

    # M = assemble(inner(w, v)*dx)

    # integrals
    aopt = l2(uref, u1)/l2(u1, u1)

    print('optimal coefficient a*: \t {0:g}'.format(aopt))

    return aopt


def plot_errors_noslip(pinrange, e_u_inf, e_u_l2, e_p_inf, e_p_l2, e_dP):

    plt.ion()
    plt.figure()
    plt.scatter(pinrange, e_u_l2)
    plt.title('relative L2 error U')
    plt.xlabel(r'$p_{in}$')
    plt.ylabel(r'$||u-u_h||_{L_2}/||u||_{L_2}$')

    plt.figure()
    plt.scatter(pinrange, e_dP)
    plt.title('relative error pressure drop')
    plt.xlabel(r'$p_{in}$')
    plt.ylabel(r'$|\Delta p - \Delta p_{ref}|/|\Delta p_{ref}|$')


def plot_errors_noslip_inflow(dRrange, umaxrange, e_u_inf, e_u_l2, e_p_inf,
                              e_p_l2, e_dP):
    plt.ion()
    for i, dR in enumerate(dRrange):
        plt.figure()
        plt.scatter(umaxrange, e_u_l2[i, :])
        plt.title(r'relative L2 error in U, $\Delta R = %g$' % dR)
        plt.xlabel(r'max $u_{in}$')
        plt.ylabel(r'$||u-u_h||_{L_2}/||u||_{L_2}$')

        plt.figure()
        plt.scatter(umaxrange, e_dP[i, :])
        plt.title(r'relative error pressure drop, $\Delta R = %g$' % dR)
        plt.xlabel(r'max $u_{in}$')
        plt.ylabel(r'$|\Delta p - \Delta p_{ref}|/|\Delta p_{ref}|$')


def plot_errors_noslip_opt_inflow(dRrange, umax, e_u_inf, e_u_l2, e_p_inf,
                                  e_p_l2, e_dP):

    plt.ion()
    plt.figure()
    plt.scatter(dRrange, e_u_l2)
    plt.title(r'relative L2 error in U, u_max opt')
    plt.xlabel(r'$\Delta R$')
    plt.xticks(dRrange)
    plt.ylabel(r'$||u-u_h||_{L_2}/||u||_{L_2}$')
    for um, x, y in zip(umax, dRrange, e_u_l2):
        ulabel = r'$u^\star_{{m}}={0:.4g}$'.format(um)
        plt.annotate(ulabel, xy=(x, y), xytext=(-42, -16),
                     textcoords='offset points')

    plt.figure()
    plt.scatter(dRrange, e_dP)
    plt.title(r'relative error pressure drop, u_max opt')
    plt.xlabel(r'$\Delta R$')
    plt.xticks(dRrange)
    plt.ylabel(r'$|\Delta p - \Delta p_{ref}|/|\Delta p_{ref}|$')
    for um, x, y in zip(umax, dRrange, e_dP):
        ulabel = r'$u^\star_{{m}}={0:.4g}$'.format(um)
        plt.annotate(ulabel, xy=(x, y), xytext=(-42, -16),
                     textcoords='offset points')

    # plt.figure()
    # plt.scatter(dRrange, umax)
    # plt.title('optimal umax for each dR-mesh')
    # plt.xlabel(r'$\Delta R$')
    # plt.ylabel(r'$u_{max}$')


def plot_errors_noslip_popt(dRrange, e_u_inf, e_u_l2, e_p_inf, e_p_l2, e_dP):

    plt.ion()
    plt.figure()
    plt.scatter(dRrange, e_u_l2, color='red')
    plt.title('relative L2 error U')
    plt.xlabel(r'$\Delta R$')
    plt.xticks(dRrange)
    plt.ylabel(r'$||u-u_h||_{L_2}/||u||_{L_2}$')

    plt.figure()
    plt.scatter(dRrange, e_dP, color='red')
    plt.title('relative error pressure drop')
    plt.xlabel(r'$\Delta R$')
    plt.xticks(dRrange)
    plt.ylabel(r'$|\Delta p - \Delta p_{ref}|/|\Delta p_{ref}|$')


def plot_errors_est_inflow(g1range, g2range, umaxrange, Rinrange, dRrange,
                           e_u_inf, e_u_l2, e_p_inf, e_p_l2, e_dP):
    ''' e_* format: [dR,g1,g2,umax,Rin] '''

    # import matplotlib.cm as cm
    import matplotlib
    from itertools import product
    matplotlib.rcParams['xtick.direction'] = 'out'
    matplotlib.rcParams['ytick.direction'] = 'out'

    # 1. calculate & plot Err(u; g1, g2) minimum f.a. (dR, umax,Rin):
    G2, G1 = np.meshgrid(g2range, g1range)
    U, R = np.meshgrid(umaxrange, Rinrange)
    plt.ion()

    g1min = []
    g2min = []
    emin_dR = []
    umin = []
    Rmin = []
    #
    # g1min2 = []
    # g2min2 = []
    # emin_dR2 = []
    #
    for i, dR in enumerate(dRrange):
        # get g1*, g2* of e_u_l2[i, :, :, j, k]
        # for j, umax in enumerate(umaxrange):
        #     for k, Rin in enumerate(Rinrange):
        g1min_loc = []
        g2min_loc = []
        h1_loc = []
        h2_loc = []
        # emin = np.zeros((len(umaxrange), len(Rinrange)))
        emin_lst = []
        for j, k in product(range(len(umaxrange)), range(len(Rinrange))):
            igmin = np.argmin(e_u_l2[i, :, :, j, k])
            h1, h2 = np.unravel_index(igmin, e_u_l2[i, :, :, j, k].shape)
            g1min_loc.append(g1range[h1])
            g2min_loc.append(g2range[h2])
            h1_loc.append(h1)
            h2_loc.append(h2)
            # emin[j, k] = e_u_l2[i, h1, h2, j, k]
            emin_lst.append(e_u_l2[i, h1, h2, j, k])

            # now g1min_loc[-1] is g1 value where e_u_l2 is minimal for
            # umaxrange[j], Rinrange[k]
            # for imshow/contour plot -> reshape(g1min_loc, (len(U*), len(R*)))

        Emin = np.array(emin_lst).reshape((len(umaxrange), len(Rinrange)))
        G1min_loc = np.array(g1min_loc).reshape((len(umaxrange),
                                                 len(Rinrange)))
        G2min_loc = np.array(g2min_loc).reshape((len(umaxrange),
                                                 len(Rinrange)))

        # find minimum Err(g1*, g2*) over (U, R)
        imin = np.argmin(Emin)
        l1, l2 = np.unravel_index(imin, Emin.shape)
        emin_dR.append(Emin[l1, l2])
        g1min.append(G1min_loc[l1, l2])
        g2min.append(G2min_loc[l1, l2])
        umin.append(umaxrange[l1])
        Rmin.append(Rinrange[l1])

        # or simpler (even more confusing)
        # jmin = np.argmin(np.array(emin_lst))
        # emin_dR2.append(emin_lst[jmin])
        # g1min2.append(g1min_loc[jmin])
        # g2min2.append(g2min_loc[jmin])

        plt.figure()
        CS = plt.contour(R, U, Emin, 20)
        plt.clabel(CS, inline=1, fontsize=10)
        plt.title('$Err_{L_2}(u;\ \gamma_1^\star, \gamma_2^\star)$ for $\Delta\
                  R = %g$' % dR)
        plt.ylabel(r'$u_{max}$')
        plt.xlabel(r'$R_{in}$')

        extent = (Rinrange.min(), Rinrange.max(),
                  umaxrange.min(), umaxrange.max())
        plt.figure()
        plt.imshow(G1min_loc[::-1, :], extent=extent, interpolation='none')
        plt.colorbar()
        plt.title(r'map of optim $\gamma_1^\star$ for $\Delta R = {0:g}$'.
                  format(dR))
        plt.ylabel(r'$u_{max}$')
        plt.xlabel(r'$R_{in}$')

        plt.figure()
        plt.imshow(G2min_loc[::-1, :], extent=extent, interpolation='none')
        plt.colorbar()
        plt.title(r'map of optim $\gamma_2^\star$ for $\Delta R = {0:g}$'.
                  format(dR))
        plt.ylabel(r'$u_{max}$')
        plt.xlabel(r'$R_{in}$')

    # print(imin == jmin)
    # print(emin_dR == emin_dR2)
    # print(g1min == g1min2)
    # print(g2min == g2min2)
    plt.figure()
    plt.scatter(dRrange, emin_dR)
    plt.xlabel(r'$\Delta R$')
    plt.ylabel(r'$Err_{L_2}(u)$')
    plt.title('Minima on each mesh over all paramters')
    for (x, y, u, r, g1, g2) in zip(dRrange, emin_dR, umin, Rmin, g1min,
                                    g2min):
        ulabel = r'$u^\star_{{m}}={0:g}$'.format(u)
        rlabel = r'$R^\star_{{in}}={0:g}$'.format(r)
        g1label = r'$\gamma_1^\star={0:.4g}$'.format(g1)
        g2label = r'$\gamma_2^\star={0:.4g}$'.format(g2)
        plt.annotate(ulabel, xy=(x, y), xytext=(4, 36),
                     textcoords='offset points')
        plt.annotate(rlabel, xy=(x, y), xytext=(4, 12),
                     textcoords='offset points')
        plt.annotate(g1label, xy=(x, y), xytext=(4, -12),
                     textcoords='offset points')
        plt.annotate(g2label, xy=(x, y), xytext=(4, -36),
                     textcoords='offset points')

    # or simpler:
    emin = []
    g1opt = []
    g2opt = []
    uopt = []
    Ropt = []
    plt.figure()
    for i, dR in enumerate(dRrange):
        emin.append(e_u_l2[i, :].min())
        imin = np.argmin(e_u_l2[i, :])
        ij = np.unravel_index(imin, e_u_l2[i, :].shape)
        g1opt.append(g1range[ij[0]])
        g2opt.append(g2range[ij[1]])
        uopt.append(umaxrange[ij[2]])
        Ropt.append(Rinrange[ij[3]])

        plt.plot(dR, emin[-1], 'bo')
        plt.annotate(r'$u_m^\star={0:g}$'.format(uopt[-1]),
                     xy=(x, y), xytext=(4, 36), textcoords='offset points')
        plt.annotate(r'$R_i^\star={0:g}$'.format(Ropt[-1]),
                     xy=(x, y), xytext=(4, 12), textcoords='offset points')
        plt.annotate(r'$\gamma_1^\star={0:g}$'.format(g1opt[-1]),
                     xy=(x, y), xytext=(4, -12), textcoords='offset points')
        plt.annotate(r'$\gamma_2^\star={0:g}$'.format(g2opt[-1]),
                     xy=(x, y), xytext=(4, -36), textcoords='offset points')
    plt.xlabel(r'$\Delta R$')
    plt.ylabel(r'$Err_{L_2}(u)$')
    plt.title('Minima on each mesh over all paramters [2]')


def plot_errors_est_inflow_uopt(g1range, g2range, Rinrange, dRrange, umaxopt,
                                e_u_inf, e_u_l2, e_p_inf, e_p_l2, e_dP):
    ''' e_* format: [dR,g1,g2,Rin] '''
    from itertools import cycle
    import matplotlib
    matplotlib.rcParams['xtick.direction'] = 'out'
    matplotlib.rcParams['ytick.direction'] = 'out'

    # 1. calculate & plot Err(u; g1, g2) minimum f.a. (dR, umax,Rin):
    plt.ion()
    #
    # g1min2 = []
    # g2min2 = []
    # emin_dR2 = []
    #
    # Simple: just plot min errors w attributes
    emin = []
    emin_dP = []
    g1opt = []
    g2opt = []
    uopt = []
    Ropt = []
    plt.figure()
    for i, dR in enumerate(dRrange):
        emin.append(e_u_l2[i, :].min())
        imin = np.argmin(e_u_l2[i, :])
        ij = np.unravel_index(imin, e_u_l2[i, :].shape)
        emin_dP.append(e_dP[i, :].flat[imin])
        g1opt.append(g1range[ij[0]])
        g2opt.append(g2range[ij[1]])
        Ropt.append(Rinrange[ij[2]])
        uopt.append(umaxopt[i, :].flat[imin])

        x, y = dR, emin[-1]
        # plt.plot(x, y, 'bo')
        plt.annotate(r'$u_m^\star={0:g}$'.format(uopt[-1]),
                     xy=(x, y), xytext=(4, 36), textcoords='offset points')
        plt.annotate(r'$R_i^\star={0:g}$'.format(Ropt[-1]),
                     xy=(x, y), xytext=(4, 12), textcoords='offset points')
        plt.annotate(r'$\gamma_1^\star={0:g}$'.format(g1opt[-1]),
                     xy=(x, y), xytext=(4, -12), textcoords='offset points')
        plt.annotate(r'$\gamma_2^\star={0:g}$'.format(g2opt[-1]),
                     xy=(x, y), xytext=(4, -36), textcoords='offset points')

    plt.scatter(dRrange, emin)
    plt.xlabel(r'$\Delta R$')
    plt.ylabel(r'$Err_{L_2}(u)$')
    plt.title('Minima on each mesh over all paramters')

    plt.figure()
    plt.scatter(dRrange, emin_dP)
    plt.xlabel(r'$\Delta R$')
    plt.ylabel(r'$\mathcal{E}_{\Delta P}$')
    plt.title('Minima on each mesh over all paramters')
    for (x, y, uo, Ro, g1o, g2o) in zip(dRrange, emin_dP, uopt, Ropt, g1opt,
                                        g2opt):
        plt.annotate(r'$u_m^\star={0:g}$'.format(uo),
                     xy=(x, y), xytext=(4, 36), textcoords='offset points')
        plt.annotate(r'$R_i^\star={0:g}$'.format(Ro),
                     xy=(x, y), xytext=(4, 12), textcoords='offset points')
        plt.annotate(r'$\gamma_1^\star={0:g}$'.format(g1o),
                     xy=(x, y), xytext=(4, -12), textcoords='offset points')
        plt.annotate(r'$\gamma_2^\star={0:g}$'.format(g2o),
                     xy=(x, y), xytext=(4, -36), textcoords='offset points')

    # for subplots or inspection of parameters
    plt.figure()
    legstr = []
    cols = cycle(['b', 'r', 'g', 'k', 'm', 'y'])
    ang = cycle(np.linspace(0, 2*np.pi, len(dRrange), endpoint=False))
    for i, dR in enumerate(dRrange):
        # get g1*, g2* of e_u_l2[i, :, :, j, k]
        # for j, umax in enumerate(umaxrange):
        #     for k, Rin in enumerate(Rinrange):
        g1min_loc = []
        g2min_loc = []
        h1_loc = []
        h2_loc = []
        # emin = np.zeros((len(umaxrange), len(Rinrange)))
        emin_lst = []
        uopt_loc = []
        for k, _ in enumerate(Rinrange):
            igmin = np.argmin(e_u_l2[i, :, :, k])
            h1, h2 = np.unravel_index(igmin, e_u_l2[i, :, :, k].shape)
            g1min_loc.append(g1range[h1])
            g2min_loc.append(g2range[h2])
            h1_loc.append(h1)
            h2_loc.append(h2)
            # emin[j, k] = e_u_l2[i, h1, h2, j, k]
            emin_lst.append(e_u_l2[i, h1, h2, k])
            uopt_loc.append(umaxopt[i, h1, h2, k])
            # now g1min_loc[-1] is g1 value where e_u_l2 is minimal for
            # umaxrange[j], Rinrange[k]
            # for imshow/contour plot -> reshape(g1min_loc, (len(U*), len(R*)))

        plt.scatter(Rinrange, emin_lst, c=cols.next())
        dtx = 80
        dty = 80
        for (x, y, g1, g2, u) in zip(Rinrange, emin_lst, g1min_loc, g2min_loc,
                                     uopt_loc):
            ulabel = r'$u^\star_{{m}}={0:g}$'.format(u)
            g1label = r'$\gamma_1^\star={0:.4g}$'.format(g1)
            g2label = r'$\gamma_2^\star={0:.4g}$'.format(g2)
            # plt.annotate(ulabel, xy=(x, y), xytext=(4, 24),
            #              textcoords='offset points')
            # plt.annotate(g1label, xy=(x, y), xytext=(4, 0),
            #              textcoords='offset points')
            # plt.annotate(g2label, xy=(x, y), xytext=(4, -24),
            #              textcoords='offset points')
            tx = dtx*np.cos(ang.next()) - dty*np.sin(ang.next())
            ty = dtx*np.sin(ang.next()) + dty*np.cos(ang.next())
            plt.annotate(ulabel + '\n' + g1label + '\n' + g2label,
                         xy=(x, y), xytext=(tx, ty),
                         textcoords='offset points',
                         bbox=dict(boxstyle='round,pad=0.5', fc='0.8',
                                   alpha=0.2),
                         arrowprops=dict(arrowstyle="->",
                                         connectionstyle="angle,angleA=0,angleB=-90,rad=10"))
            #                            # connectionstyle='arc3,rad=0'))

            # ax.annotate('angle', xy=(3.5, -1), xycoords='data',
            #             xytext=(-70, -60), textcoords='offset points',
            #             size=20,
            #             bbox=dict(boxstyle="round4,pad=.5", fc="0.8"),
            #             arrowprops=dict(arrowstyle="->",
            #                             connectionstyle="angle,angleA=0,angleB=-90,rad=10"),
            #             )

        legstr.append(r'$\Delta R = {0:g}$'.format(dR))
    plt.xlabel(r'$R_{in}$')
    plt.ylabel(r'$Err_{L_2}(u)$')
    tstr = r'E min over $(\gamma_1, \gamma_2, u_{{max}},R _{{in}})$'
    plt.title(tstr.format(dR))
    plt.legend(legstr)


def plot_errors(g1range, g2range, pinrange, e_u_inf, e_u_l2, e_p_inf, e_p_l2,
                e_dP):
    # import matplotlib.cm as cm
    import matplotlib
    matplotlib.rcParams['xtick.direction'] = 'out'
    matplotlib.rcParams['ytick.direction'] = 'out'

    X, Y = np.meshgrid(g2range, g1range)
    plt.ion()
    # XXX THIS MAKES NO SENSE AT ALL
    # for i, pin in enumerate(pinrange):
    #     plt.figure()
    #     CS = plt.contour(X, Y, e_dP[:, :, i], 20)
    #     plt.clabel(CS, inline=1, fontsize=10)
    #     plt.title(r'$e(\Delta P)_\infty$ for $p_{in}[%i] = %.2f$' % (i, pin))
    #     plt.xlabel(r'$\gamma_1$')
    #     plt.ylabel(r'$\gamma_2$')

    for i, pin in enumerate(pinrange):
        plt.figure()
        CS = plt.contour(X, Y, e_u_l2[i, :, :], 20)
        plt.clabel(CS, inline=1, fontsize=10)
        plt.title(r'$e(u)_{L_2}$ for $p_{in}[%i] = %.2f$' % (i, pin))
        plt.xlabel(r'$\gamma_2$ (coarctation)')
        plt.ylabel(r'$\gamma_1$ (pipe)')

    # SCATTER PLOT min e_u_inf(p_in)
    # g1, g2 coordinate of minimum error per p_in
    g1min = []
    g2min = []
    for i in range(len(pinrange)):
        # TODO: could annotate plot directly in this loop
        imin = np.argmin(e_u_l2[i, :, :])
        h1, h2 = np.unravel_index(imin, e_u_l2[i, :, :].shape)
        # i1.append(h1)
        # i2.append(h2)
        g1min.append(g1range[h1])
        g2min.append(g2range[h2])
        labels = ['({0:.2g}, {1:.2g})'.format(g1, g2) for (g1, g2) in
                  zip(g1min, g2min)]

    emin = np.min(np.min(e_u_l2, axis=1), axis=1)

    plt.figure()
    plt.ion()
    plt.scatter(pinrange, emin)
    for label, x, y in zip(labels, pinrange, emin):
        plt.annotate(label, xy=(x, y), xytext=(-4, 4),
                     textcoords='offset points')
    plt.xlabel(r'$p_{in}$')
    plt.ylabel(r'$||u-u_h||_{L_2}/||u||_{L_2}$')
    plt.title('Plot min err w.r.t. (g1, g2, p_in=const) over p_in')
    plt.legend((r'$(\gamma_1, \gamma_2)$',))

    # SCATTER PLOT min e_dP(p_in)
    # PRESSURE DIFFERENCE IMPOSED AS BC, DOESN'T DEPEND ON GAMMA1/2!
    g1min = []
    g2min = []
    for i in range(len(pinrange)):
        # TODO: could annotate plot directly in this loop
        imin = np.argmin(e_dP[i, :, :])
        h1, h2 = np.unravel_index(imin, e_dP[i, :, :].shape)
        # i1.append(h1)
        # i2.append(h2)
        g1min.append(g1range[h1])
        g2min.append(g2range[h2])
        labels = ['({0:.2g}, {1:.2g})'.format(g1, g2) for (g1, g2) in
                  zip(g1min, g2min)]

    emin = np.min(np.min(e_dP, axis=1), axis=1)

    plt.figure()
    plt.ion()
    plt.scatter(pinrange, emin)
    for label, x, y in zip(labels, pinrange, emin):
        plt.annotate(label, xy=(x, y), xytext=(-4, 4),
                     textcoords='offset points')
    plt.xlabel(r'$p_{in}$')
    plt.ylabel(r'$|\Delta p - \Delta p_{ref}|/|\Delta p_{ref}|$')
    plt.title('Plot min err w.r.t. (g1, g2, p_in=const) over p_in')
    plt.legend((r'$(\gamma_1, \gamma_2)$',))


def plot_error_compare(dRrange, err_u_l2_ns, err_dP_ns, err_u_l2_gm,
                       err_dP_gm, err_dP_STE):
    emin_u_gm = []
    emin_dP_gm = []
    for i, dR in enumerate(dRrange):
        emin_u_gm.append(err_u_l2_gm[i, :].min())
        imin = np.argmin(err_u_l2_gm[i, :])
        # ij = np.unravel_index(imin, e_u_l2[i, :].shape)
        emin_dP_gm.append(err_dP_gm[i, :].flat[imin])
        # g1opt.append(g1range[ij[0]])
        # g2opt.append(g2range[ij[1]])
        # Ropt.append(Rinrange[ij[2]])
        # uopt.append(umaxopt[i, :].flat[imin])

    plt.ion()
    if False:
        plt.figure()
        plt.scatter(dRrange, err_u_l2_ns, color='red')
        plt.scatter(dRrange, emin_u_gm, color='blue')
        plt.title(r'$L_2$ error norm of velocity')
        plt.xlabel('$\Delta R/R$')
        plt.ylabel(r'$\mathrm{err}_{L_2}(u)$')
        plt.legend(('no-slip', r'$\gamma$-slip'), loc=2)

        plt.figure()
        plt.scatter(dRrange, err_dP_ns, color='red')
        plt.scatter(dRrange, emin_dP_gm, color='blue')
        plt.title(r'relative error of pressure drop $\Delta p$')
        plt.xlabel('$\Delta R/R$')
        plt.ylabel(r'$\mathcal{E}_{\Delta p}$')
        plt.legend(('no-slip', r'$\gamma$-slip'), loc=2)

    barwidth = 0.35
    opacity = 0.4
    index = np.arange(len(emin_u_gm))
    plt.figure()
    plt.bar(index, err_u_l2_ns, barwidth, label='no-slip', color='blue',
            alpha=opacity)
    plt.bar(index + barwidth, emin_u_gm, barwidth, label=r'$\gamma$-slip',
            color='red', alpha=opacity)
    plt.xlabel(r'$\Delta R/R$')
    plt.ylabel(r'$\mathrm{err}_{L_2}(u)$')
    plt.legend()
    plt.xticks(index + barwidth, dRrange)

    barwidth = 0.25
    plt.figure()
    plt.bar(index, emin_dP_gm, barwidth, label=r'$\gamma$-slip', color='blue',
            alpha=opacity)
    plt.bar(index + barwidth, err_dP_ns, barwidth, label='no-slip',
            color='red', alpha=opacity)
    plt.bar(index + 2*barwidth, err_dP_STE, barwidth, label=r'STE int',
            color='green', alpha=opacity)
    plt.xlabel(r'$\Delta R/R$')
    plt.ylabel(r'$\mathcal{E}_{\Delta p}$')
    plt.legend(loc=2)
    plt.xticks(index + 1.5*barwidth, dRrange)


def plot_errors_popt(g1range, g2range, dRrange, pin, e_u_inf, e_u_l2, e_p_inf,
                     e_p_l2, e_dP):
    # import matplotlib.cm as cm
    import matplotlib
    matplotlib.rcParams['xtick.direction'] = 'out'
    matplotlib.rcParams['ytick.direction'] = 'out'

    X, Y = np.meshgrid(g2range, g1range)
    plt.ion()
    extent = [g2range[0], g2range[1], g1range[0], g1range[1]]
    for i, dR in enumerate(dRrange):
        plt.figure()
        CS = plt.contour(X, Y, e_u_l2[i, :, :], 20)
        plt.clabel(CS, inline=1, fontsize=10)
        plt.title(r'$e(u)_{{L_2}}$ for $\Delta R = {0}$'.format(dR))
        plt.xlabel(r'$\gamma_2$ (coarctation)')
        plt.ylabel(r'$\gamma_1$ (pipe)')
        plt.figure()
        plt.imshow(e_u_l2[i, ::-1, :], interpolation='none', extent=extent)
        plt.colorbar(label=r'$err_{L_2}(u)$')
        plt.title(r'$err_{{L_2}}(u)$ for $\Delta R = {0}$'.format(dR))
        plt.xlabel(r'$\gamma_2$ (coarctation)')
        plt.ylabel(r'$\gamma_1$ (pipe)')
        #
        plt.figure()
        # CS = plt.contour(X, Y, pin[:, :, i], 20)
        # plt.clabel(CS, inline=1, fontsize=10)
        plt.imshow(pin[i, ::-1, :], interpolation='none', extent=extent)
        plt.colorbar(label=r'$p^*$')
        plt.title(r'$p^*$ for $\Delta R = {0}$'.format(dR))
        plt.xlabel(r'$\gamma_2$ (coarctation)')
        plt.ylabel(r'$\gamma_1$ (pipe)')

    # SCATTER PLOT min e_u_inf(dR)
    # g1, g2 coordinate of minimum error per dR
    g1min = []
    g2min = []
    pmin = []
    for i in range(len(dRrange)):
        # TODO: could annotate plot directly in this loop
        imin = np.argmin(e_u_l2[i, :, :])
        h1, h2 = np.unravel_index(imin, e_u_l2[i, :, :].shape)
        # i1.append(h1)
        # i2.append(h2)
        g1min.append(g1range[h1])
        g2min.append(g2range[h2])
        pmin.append(pin[i, h1, h2])
        glabels = ['({0:.2g}, {1:.2g})'.format(g1, g2) for (g1, g2) in
                   zip(g1min, g2min)]
        plabels = [r'$p^*={0:.2g}$'.format(p) for p in pmin]

    emin = np.min(np.min(e_u_l2, axis=1), axis=1)

    plt.figure()
    plt.ion()
    plt.scatter(dRrange, emin)
    for glabel, plabel, x, y in zip(glabels, plabels, dRrange, emin):
        plt.annotate(glabel, xy=(x, y), xytext=(-4, 4),
                     textcoords='offset points')
        plt.annotate(plabel, xy=(x, y), xytext=(-16, -12),
                     textcoords='offset points')
    plt.xlabel(r'$\Delta R$')
    plt.xticks(dRrange)
    plt.ylabel(r'$||u-u_h||_{L_2}/||u||_{L_2}$')
    plt.title(r'Plot min err w.r.t. (g1, g2) over $\Delta R$')
    plt.legend((r'$(\gamma_1, \gamma_2)$',))

    # SCATTER PLOT min e_dP(p_in)
    # PRESSURE DIFFERENCE IMPOSED AS BC, DOESN'T DEPEND ON GAMMA1/2!
    # g1min = []
    # g2min = []
    # for i in range(len(pinrange)):
    #     # TODO: could annotate plot directly in this loop
    #     imin = np.argmin(e_dP[:, :, i])
    #     h1, h2 = np.unravel_index(imin, e_dP[:, :, i].shape)
    #     # i1.append(h1)
    #     # i2.append(h2)
    #     g1min.append(g1range[h1])
    #     g2min.append(g2range[h2])
    #     labels = ['({0:.2g}, {1:.2g})'.format(g1, g2) for (g1, g2) in
    #               zip(g1min, g2min)]

    # emin = np.min(np.min(e_dP, axis=0), axis=0)

    # plt.figure()
    # plt.ion()
    # plt.scatter(pinrange, emin)
    # for label, x, y in zip(labels, pinrange, emin):
    #     plt.annotate(label, xy=(x, y), xytext=(-4, 4),
    #                  textcoords='offset points')
    # plt.xlabel(r'$p_{in}$')
    # plt.ylabel(r'$|\Delta p - \Delta p_{ref}|/|\Delta p_{ref}|$')
    # plt.title('Plot min err w.r.t. (g1, g2, p_in=const) over p_in')
    # plt.legend((r'$(\gamma_1, \gamma_2)$',))


def load_npz(npzfile):

    A = np.load(npzfile)
    g1range = A['g1range']
    g2range = A['g2range']
    Rinrange = A['Rinrange']
    dRrange = A['dRrange']
    e_u_inf = A['e_u_inf']
    e_u_l2 = A['e_u_l2']
    e_p_inf = A['e_p_inf']
    e_p_l2 = A['e_p_l2']
    e_dP = A['e_dP']
    try:
        umaxopt = A['umaxopt']
    except:
        umaxopt = None
    try:
        e_dPmeas = A['e_dPmeas']
    except:
        e_dPmeas = None

    return (g1range, g2range, dRrange, umaxopt, Rinrange, e_u_inf, e_u_l2,
            e_p_inf, e_p_l2, e_dP, e_dPmeas)


if __name__ == '__main__':
    # (g1range, g2range, dRrange, pin, e_u_inf, e_u_l2, e_p_inf, e_p_l2, e_dP),\
    #     (uref, u) = aorta_narrowed_est_popt(recalc=False, mesh_reflvl=0)

    # np.savez('estim_g1g2dR_popt.npz', g1range=g1range, g2range=g2range,
    #          pin=pin, dRrange=dRrange, e_u_inf=e_u_inf, e_u_l2=e_u_l2,
    #          e_p_inf=e_p_inf, e_p_l2=e_p_l2, e_dP=e_dP)

    # plot_errors_popt(g1range, g2range, dRrange, pin, e_u_inf, e_u_l2, e_p_inf,
    #                  e_p_l2, e_dP)

    # plot_errors(g1range, g2range, pinrange, e_u_inf, e_u_l2, e_p_inf, e_p_l2,
    #             e_dP)

    # (g1range, g2range, umaxrange, Rinrange, dRrange, e_u_inf, e_u_l2, e_p_inf,
    #  e_p_l2, e_dP), (uref, u) = \
    #     aorta_narrowed_est_inflow(mesh_reflvl=0)

    # plot_errors_est_inflow(g1range, g2range, umaxrange, Rinrange, dRrange,
    #                        e_u_inf, e_u_l2, e_p_inf, e_p_l2, e_dP)

    # (g1range, g2range, Rinrange, dRrange, umaxopt, e_u_inf, e_u_l2, e_p_inf,
    #     e_p_l2, e_dP), (uref, u) = \
    #     aorta_narrowed_est_inflow_uopt(mesh_reflvl=0)

    # np.savez('estim_inflow_uopt1.npz', g1range=g1range, g2range=g2range,
    #          Rinrange=Rinrange, dRrange=dRrange, umaxopt=umaxopt,
    #          e_u_inf=e_u_inf, e_u_l2=e_u_l2, e_p_inf=e_p_inf, e_p_l2=e_p_l2,
    #          e_dP=e_dP)

    # plot_errors_est_inflow_uopt(g1range, g2range, Rinrange, dRrange, umaxopt,
    #                             e_u_inf, e_u_l2, e_p_inf, e_p_l2, e_dP)

    # u1, p1, dP1 = solve_aorta(lvl=1)

    # ###################################################################### #
    #                            NO SLIP
    # ###################################################################### #

    # (pinrange, dRrange, e_u_inf, e_u_l2, e_p_inf, e_p_l2, e_dP), \
    #     (uref, u) = aorta_narrowed_est_noslip()
    # np.savez('estim_3610p_noslip.npz', g1range=g1range, g2range=g2range,
    #          pinrange=pinrange, e_u_inf=e_u_inf, e_u_l2=e_u_l2,
    #          e_p_inf=e_p_inf, e_p_l2=e_p_l2, e_dP=e_dP)
    # plot_errors_noslip(pinrange, e_u_inf, e_u_l2, e_p_inf, e_p_l2, e_dP)

    # (dRrange, e_u_inf, e_u_l2, e_p_inf, e_p_l2, e_dP), (uref, u) = \
    #     aorta_narrowed_est_noslip_popt(recalc=False, mesh_reflvl=1)
    # plot_errors_noslip_popt(dRrange, e_u_inf, e_u_l2, e_p_inf, e_p_l2, e_dP)

    # a_opt = optimal_coefficient(uref, u)

    # _, (uref, u) = aorta_narrowed_est()
    # _, (uref, u) = aorta_narrowed_est_noslip()
    # g1range, g2range, pinrange, e_u_inf, e_u_l2, e_p_inf, e_p_l2, e_dP = \
    #     load_npz('estim_3610p.npz')

    # (dRrange, umaxrange, e_u_inf, e_u_l2, e_p_inf, e_p_l2, e_dP), (u, uref) = \
    #     aorta_narrowed_est_noslip_inflow(mesh_reflvl=0)
    # plot_errors_noslip_inflow(dRrange, umaxrange, e_u_inf, e_u_l2, e_p_inf,
    #                           e_p_l2, e_dP)

    # (dRrange, umax, e_u_inf, e_u_l2, e_p_inf, e_p_l2, e_dP), (u, uref) = \
    #     aorta_narrowed_est_noslip_opt_inflow(mesh_reflvl=0)
    # plot_errors_noslip_opt_inflow(dRrange, umax, e_u_inf, e_u_l2, e_p_inf,
    #                               e_p_l2, e_dP)

    # ###################################################################### #
    #                            STE INT
    # ###################################################################### #
    # aorta_ref_STEint(mesh_reflvl=0)
    # err_dP_STE_ref, err_dP_STE, _ = aorta_narrowed_STEint(mesh_reflvl=0)

    # B = load_npz('estim_inflow_uopt1.npz')
    # (_, _, dRrange, _, _, _, err_u_l2_gm, _, _, err_dP_gm) = B

    # (_, _, _, err_u_l2_ns, _, _, err_dP_ns), (_, _) = \
    #     aorta_narrowed_est_noslip_opt_inflow(mesh_reflvl=0)

    # plot_error_compare(dRrange, err_u_l2_ns, err_dP_ns, err_u_l2_gm, err_dP_gm,
    #                    err_dP_STE)

    # ###################################################################### #
    #                      MEASRMT NS vs GM vs STEint                        #
    # ###################################################################### #
    # for h in [0.05, 0.025, 0.01]:
    #     (g1range, g2range, Rinrange, dRrange, umaxopt, e_u_inf, e_u_l2,
    #      e_p_inf, e_p_l2, e_dP, e_dPmeas), (uref, u) = \
    #         aorta_narrowed_est_inflow_uopt_measrmt(h1=h)

    #     np.savez('estim_inflow_uopt_meas_h{0:g}.npz'.format(h),
    #              g1range=g1range, g2range=g2range, Rinrange=Rinrange,
    #              dRrange=dRrange, umaxopt=umaxopt, e_u_inf=e_u_inf,
    #              e_u_l2=e_u_l2, e_p_inf=e_p_inf, e_p_l2=e_p_l2, e_dP=e_dP,
    #              e_dPmeas=e_dPmeas)
    (g1range, g2range, Rinrange, dRrange, umaxopt, e_u_inf, e_u_l2,
     e_p_inf, e_p_l2, e_dP, e_dPmeas), (uref, u) = \
        aorta_narrowed_est_inflow_uopt_measrmt()

    np.savez('estim_inflow_uopt_meas2.npz',
             g1range=g1range, g2range=g2range, Rinrange=Rinrange,
             dRrange=dRrange, umaxopt=umaxopt, e_u_inf=e_u_inf,
             e_u_l2=e_u_l2, e_p_inf=e_p_inf, e_p_l2=e_p_l2, e_dP=e_dP,
             e_dPmeas=e_dPmeas)

    B = load_npz('estim_inflow_uopt_meas2.npz')
    (g1range, g2range, dRrange, umaxopt, Rinrange, e_u_inf, err_u_l2_gm,
     e_p_inf, e_p_l2, err_dP_gm, err_dPmeas_gm) = B

    plot_errors_est_inflow_uopt(g1range, g2range, Rinrange, dRrange,
                                umaxopt, e_u_inf, e_u_l2, e_p_inf, e_p_l2,
                                e_dP)

    err_dP_STE_ref, err_dP_STE, err_dP_STE_meas = \
        aorta_narrowed_STEint_measrmt()

    (_, _, _, err_u_l2_ns, _, _, err_dP_ns, err_dPmeas_ns), (_, _) = \
        aorta_narrowed_est_noslip_uopt_inflow_measrmt()

    plot_error_compare(dRrange, err_u_l2_ns, err_dP_ns, err_u_l2_gm, err_dP_gm,
                       err_dP_STE)
    plt.title('Errors w.r.t. to reference solution')
