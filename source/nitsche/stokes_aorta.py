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

import platform
import re
import shutil
import os
import sys

from functions.utils import on_cluster


def petsc_args():
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


def getLS(mesh, beta=10, solver='lu', elements='TH', gamma=0.0, prec='',
          wall='noslip', uwall=0.0, bctype='nitsche', symmetric=False,
          block_assemble=False, plots=False, inlet='parabola', Rref=None,
          umax=1., mu=0.01):
    ''' Stokes solver for 3D pipe flow problems.
        input:
            mesh
            wall:       none, noslip, slip (free), navierslip (friction)
            uwall:      velocity of moving wall (noslip)
            bctype:     dirichlet, nitsche
            beta:       penalty for Nitsche BCs
            solver:     solver for linear system (e.g. mumps, lu or krylov
                        space method with suitable preconditioner)
            prec:       preconditioner for krylov space method
            poiseuille: bool, compare with exact poiseuille channel solution
            plots:      bool, plot solution
            inlet:      inlet velocity profile, parabola or const
            Rref:       reference radius for inlet parabolic profile & bcs
            umax:       max. velocity for inflow
        output:
            u1, p1      solution
            W.dim()     number of DOFs
    '''
    from functions.inout import read_mesh
    mesh, _, bnds = read_mesh(mesh)

    ndim = mesh.geometry().dim()

    k = Constant(mu)
    beta = Constant(beta)
    if type(gamma) in [float, int]:
        gamma = Constant(gamma)

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

    # boundary numbering
    # 1   inlet
    bnd_in = 1
    # 2   outlet
    # bnd_out = 2
    # 3   wall
    bnd_w = 3
    # 4   slip/symmetry
    bnd_s = 4

    ds = Measure('ds', domain=mesh, subdomain_data=bnds)

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

    if elements == 'Mini':
        parameters['krylov_solver']['monitor_convergence'] = False
        inflow = project(inflow, W.sub(0).collapse())

    # outflow: du/dn + pn = 0 on ds(2)

    # Define variational problem

    n = FacetNormal(mesh)
    h = CellSize(mesh)

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

        A = assemble(a1)
        b = assemble(L1)
        [bc.apply(A, b) for bc in bcs]

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
        a11 = Constant(0)*p*q*dx

        L0 = f(v)
        L1 = Constant(0)*q*dx

        # XXX TODO: SIMPLIFY FORMS --> tt = (I-nn), etc
        # inflow
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

    return A, b, W, ds  # return ds for pressure jump integration


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


def solveLS_numpy(A, b, W):
    A = sp.csr_matrix(A)

    # invert A
    Ainv = sl.inv(A)

    pass


def solveLS_PETSc(A, P, b, W):
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
            pc.setFieldSplitSchurPreType(PETSc.PC.SchurPreType.USER, P)
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


def solve_aorta2d():

    R = 1.0
    Ri = 1.0

    gamma = 2.*Ri/(Ri**2 - R**2)
    gamma = 0
    beta = 20.
    umax = 3.

    mesh_reflvl = 1
    mesh_file = 'coarc2d_r%d.h5' % mesh_reflvl

    if on_cluster():
        # running on NLHPC cluster
        mfile = '/home/dnolte/fenics/nitsche/meshes/' + mesh_file
        mesh = '/dev/shm/' + mesh_file
        shutil.copy(mfile, '/dev/shm/')
    else:
        # running on local machine
        mesh = 'meshes/' + mesh_file

    def uwall(umax, R, y):
        return -umax/R**2*(y**2 - R**2)

    tic()
    u1, p1, dof = stokes(mesh,
                         wall='noslip',
                         uwall=uwall(umax, R, Ri),  # noslip only
                         bctype='nitsche',
                         elements='Mini',
                         symmetric=True,
                         gamma=gamma,            # navier-slip only
                         beta=beta,              # Nitsche only
                         solver='krylov',
                         prec='schur',
                         plots=True,
                         inlet='parabola',  # parabola, const
                         Rref=R,
                         umax=umax
                         )

    # f1 = File('results/pipe4s_r%d.pvd' % mesh_reflvl)
    # f1 << u1, p1

    if MPI.COMM_WORLD.rank == 0:
        print('time elapsed:\t %fs' % toc())

    if on_cluster():
        # delete mesh from /dev/shm when done
        os.remove(mesh)


def solve_aorta(mesh, R, meshfile):
    R = 1.0
    Ri = 1.0

    gamma = 2.*Ri/(Ri**2 - R**2)
    beta = 20.
    umax = 3.

    mesh_reflvl = 1
    mesh_file = 'coarc2d_r%d.h5' % mesh_reflvl

    mesh = prep_mesh(mesh_file)

    tic()
    u1, p1, dof = stokes(mesh,
                         wall='noslip',
                         uwall=uwall(umax, R, Ri),  # noslip only
                         bctype='nitsche',
                         elements='Mini',
                         symmetric=True,
                         gamma=gamma,            # navier-slip only
                         beta=beta,              # Nitsche only
                         solver='krylov',
                         prec='schur',
                         plots=True,
                         inlet='parabola',  # parabola, const
                         Rref=R,
                         umax=umax
                         )

    # f1 = File('results/pipe4s_r%d.pvd' % mesh_reflvl)
    # f1 << u1, p1

    if MPI.COMM_WORLD.rank == 0:
        print('time elapsed:\t %fs' % toc())

    if on_cluster():
        # delete mesh from /dev/shm when done
        os.remove(mesh)


def aorta_narrowed():
    # from itertools import product

    set_log_level(ERROR)
    R1 = 1.0
    R2 = 1.1
    dR = 0.1   # R2 - R1

    mu = 0.01
    # gamma fixed for inlet radii
    gamma1 = -2.*mu*R1/(2*R1*dR + dR**2)
    # gamma as function of coordinates: r = y = x[1].
    # gamma2 = Expression('-2.0*mu*x[1]/(2.0*x[1]*dR + dR*dR)', mu=mu, dR=dR)
    gamma2 = Expression('2.0*mu*x[1]/(x[1]*x[1] - R*R)', mu=mu, R=R2)

    # geometry
    xi = 1.0
    L0 = 2.0
    scale_gm = 1.0

    # TODO: CHOOSE.
    gamma_split = Expression('x[0] < xi || x[0] > xi+L ? \
                             2.0*mu*x[1]/(x[1]*x[1] - R*R) :\
                             A*2.0*mu*x[1]/(x[1]*x[1] - R*R)',
                             mu=mu, R=R2, xi=xi, L=L0, A=scale_gm)

    # conditional!??
    gamma_split = Expression('x[0] < xi || x[0] > xi+L ? \
                             2.0*mu*R1/(R1*R1 - R*R) :\
                             A*2.0*mu*R1/(R1*R1 - R*R)',
                             mu=mu, R1=R1, R=R2, xi=xi, L=L0, A=scale_gm)

    # gamma_split = Expression('x[0] < xi || x[0] > xi+L ? \
    #                          -2.0*mu*x[1]/(2.0*x[1]*dR + dR*dR) :\
    #                          -A*2.0*mu*x[1]/(2.0*x[1]*dR + dR*dR)',
    #                          mu=mu, dR=dR, xi=xi, L=L0, A=scale_gm)
    beta = 20.
    umax = 2.

    ELE = 'Mini'

    mesh_reflvls = [0, 1]
    gammas = [gamma1, gamma2]
    mesh_reflvls = [1]
    gammas = [gamma2]

    scale_gm = np.arange(0.0, 3.0, 0.1)

    # TODO: MAKE TRUE LOOP ONLY OVER GAMMAS; DON'T REASSEMBLE EVERYTHING EACH
    # TIME!

    # TODO: improve algorithm & variable names
    # for gamma, mesh_reflvl in product(gammas, mesh_reflvls):
    dP = []
    errLinf_p = []
    errLinf_u = []
    errL2_p = []
    errL2_u = []
    for mesh_reflvl in mesh_reflvls:
        mesh_file_ref = 'coarc2d+dR_f0.6_r%d.h5' % mesh_reflvl
        mesh_file_gm = 'coarc2d_f0.6_r%d.h5' % mesh_reflvl
        mesh_ref = prep_mesh(mesh_file_ref)
        mesh_gm = prep_mesh(mesh_file_gm)

        # reference case
        A, b, W, ds = getLS(mesh=mesh_ref,
                            wall='noslip',
                            uwall=0.0,
                            # uwall=uwall(umax, R2, R1),  # noslip only
                            bctype='nitsche',
                            beta=beta,              # Nitsche only
                            gamma=None,            # navier-slip only
                            elements=ELE,
                            symmetric=True,
                            inlet='parabola',  # parabola, const
                            Rref=R2,        # true reference radius inlet
                            umax=umax,      # inlet velocity
                            mu=mu
                            )
        Wref = W
        uref, pref = solveLS(A, b, W, 'mumps', plots=False)

        # pressure jump
        dPref = assemble(pref*ds(1) - pref*ds(2))
        print('Pressure jump dP_ref: %f' % dPref)

        A, b, W, ds = getLS(mesh=mesh_gm,
                            wall='noslip',
                            uwall=0.0,
                            # uwall=uwall(umax, R2, R1),  # noslip only
                            bctype='nitsche',
                            beta=beta,              # Nitsche only
                            gamma=None,            # navier-slip only
                            elements=ELE,
                            symmetric=True,
                            inlet='parabola',  # parabola, const
                            Rref=R1,        # true reference radius inlet
                            umax=umax,      # inlet velocity
                            mu=mu
                            )
        uref2, pref2 = solveLS(A, b, W, 'mumps', plots=False)

        # pressure jump
        dPref2 = assemble(pref2*ds(1) - pref2*ds(2))
        print('Pressure jump dP_ref2: %f' % dPref2)

        urefi = uref
        if ELE == 'Mini':   # XXX DO THIS ONLY ONCE
            from functions.inout import read_mesh
            mesh, _, _ = read_mesh(mesh_ref)
            V = VectorFunctionSpace(mesh, "CG", 1)
            urefi = interpolate(urefi, V)
        parameters['krylov_solver']['monitor_convergence'] = False
        urefi = project(urefi, W.sub(0).collapse())
        prefi = interpolate(pref, W.sub(1).collapse())

        unorm_l2 = norm(urefi.vector(), 'l2')
        unorm_inf = norm(urefi.vector(), 'linf')
        pnorm_l2 = norm(prefi.vector(), 'l2')
        pnorm_inf = norm(prefi.vector(), 'linf')

        err_dPref = abs(dPref - dPref2)/dPref
        errL2_uref = norm(urefi.vector() - uref2.vector(), 'l2')/unorm_l2
        errLinf_uref = norm(urefi.vector() - uref2.vector(), 'linf')/unorm_inf
        errL2_pref = norm(prefi.vector() - pref2.vector(), 'l2')/pnorm_l2
        errLinf_pref = norm(prefi.vector() - pref2.vector(), 'linf')/pnorm_inf
        err_ref = (err_dPref, errL2_uref, errLinf_uref, errL2_pref, errLinf_pref)

        sol = []

        for i, gi in enumerate(scale_gm):

            tic()

            gamma_split.A = gi
            A, b, W, ds = getLS(mesh=mesh_gm,
                                wall='navierslip',
                                gamma=gamma_split,         # navier-slip only
                                uwall=None,
                                # uwall=uwall(umax, R2, R1),  # noslip only
                                bctype='nitsche',
                                beta=beta,              # Nitsche only
                                elements=ELE,
                                symmetric=True,
                                inlet='parabola',  # parabola, const
                                Rref=R2,        # true reference radius inlet
                                umax=umax,      # inlet velocity
                                mu=mu
                                )
            # Wn.append(W)
            u, p = solveLS(A, b, W, 'mumps', plots=False)
            sol.append((u, p))

            # https://bitbucket.org/fenics-project/dolfin/issues/489/wrong-interpolation-for-enriched-element#comment-16480392
            ''' This effectively disables interpolation to enriched space,
            projection from enriched to another mesh and using expressions and
            constants in Dirichlet BCs on enriched space. The rest (i.e.
            projection to enriched, interpolation from enriched to non-enriched
            on the same mesh and Functions in Dirichlet values on enriched
            space) should be working correctly.
            '''
            urefi = uref
            if ELE == 'Mini':   # XXX DO THIS ONLY ONCE
                from functions.inout import read_mesh
                mesh, _, _ = read_mesh(mesh_ref)
                V = VectorFunctionSpace(mesh, "CG", 1)
                urefi = interpolate(urefi, V)
            parameters['krylov_solver']['monitor_convergence'] = False
            urefi = project(urefi, W.sub(0).collapse())
            prefi = interpolate(pref, W.sub(1).collapse())

            unorm_l2 = norm(urefi.vector(), 'l2')
            unorm_inf = norm(urefi.vector(), 'linf')
            pnorm_l2 = norm(prefi.vector(), 'l2')
            pnorm_inf = norm(prefi.vector(), 'linf')

            # plot(u - urefi)

            dPi = assemble(p*ds(1) - p*ds(2))
            dP.append(dPi)

            # TODO norm(du) ??
            errL2_u.append(norm(urefi.vector() - u.vector(), 'l2'))
            errLinf_u.append(norm(urefi.vector() - u.vector(), 'linf'))

            errL2_p.append(norm(prefi.vector() - p.vector(), 'l2'))
            errLinf_p.append(norm(prefi.vector() - p.vector(), 'linf'))

            # du = project(u - urefi, W.sub(0).collapse())
            # plot(du)
            # f1 = File('results/coarc2d/error.pvd')
            # f1 << du
            # f2 = File('results/coarc2d/u_noslip.pvd')
            # f2 << u2
            # f3 = File('results/coarc2d/u_gamma.pvd')
            # f3 << u1
            # f4 = File('results/coarc2d/p_noslip.pvd')
            # f4 << p2
            # f5 = File('results/coarc2d/p_gamma.pvd')
            # f5 << p1

            if MPI.COMM_WORLD.rank == 0:
                print('\nGamma scaling factor: %.3f' % gi)
                #     # ('Expression' if 'expression' in str(type(gamma)) else
                #     #  str(gamma)), gi)
                print('Refine level %g' % mesh_reflvl)
                print('time elapsed:\t %fs' % toc())
                print('error @ %g DOFs' % W.dim())
                print('Pressure jump dP  (dPref): %f \t (%f)' % (dPi, dPref))
                print('relative error:\t %f' % ((dPref - dPi)/dPref))
                # print('U L2:\t\t %f' % errL2_u[-1])
                print('U L2 rel:\t %f' % (errL2_u[-1]/unorm_l2))
                print('U Linf:\t\t %f' % errLinf_u[-1])
                print('U Linf rel:\t %f' % (errLinf_u[-1]/unorm_inf))
                # print('P L2:\t\t %f' % errL2_p[-1])
                print('P L2 rel:\t %f' % (errL2_p[-1]/pnorm_l2))
                print('P Linf:\t\t %f' % errLinf_p[-1])
                print('P Linf rel:\t %f' % (errLinf_p[-1]/pnorm_inf))

        if on_cluster():
            # delete mesh from /dev/shm when done
            os.remove(mesh_ref)
            os.remove(mesh_gm)

    dP = np.array(dP)

    return (scale_gm, abs((dP-dPref))/dPref, np.array(errLinf_u)/unorm_inf,
            np.array(errL2_u)/unorm_l2, np.array(errLinf_p)/pnorm_inf,
            np.array(errL2_p)/pnorm_l2, err_ref)


def plot_errors(gm, dPrel, e_uinf, e_ul2, e_pinf, e_pl2, err_ref):
    plt.figure()
    plt.ion()
    plt.plot(gm, dPrel, 'o')
    plt.xlabel(r'$\gamma$ scaling factor (-)')
    plt.ylabel('rel. error pressure jump')

    plt.figure()
    plt.ion()
    plt.plot(gm, dPrel, '^-')
    plt.plot(gm, e_uinf, 'o', gm, e_ul2, 'v', gm, e_pinf, 's', gm, e_pl2, 'd')
    plt.ylabel(r'relative Error')
    plt.xlabel(r'$\gamma$ scaling factor (-)')
    plt.grid()

    plt.plot(1.0, err_ref[0], 'x', 1.0, err_ref[1], 'x', 1.0, err_ref[2], 'x',
             1.0, err_ref[3], 'x', 1.0, err_ref[4], 'x', ms=14, mew=2)

    plt.legend((r'$(\Delta P - \Delta P_{ref})/\Delta P_{ref}$',
                r'$||u-u_h||_{L_2}/||u||_{L_2}$',
                r'$||u-u_h||_{\infty}/||u||_{\infty}$',
                r'$||p-p_h||_{L_2}/||p||_{L_2}$',
                r'$||p-p_h||_{\infty}/||p||_{\infty}$',
                r'$e(\Delta P)$ ref/no-slip',
                r'$e(u_h)_{L_2}$ ref/no-slip',
                r'$e(u_h)_{\infty}$ ref/no-slip',
                r'$e(p_h)_{L_2}$ ref/no-slip',
                r'$e(p_h)_{\infty}$ ref/no-slip'))


if __name__ == '__main__':
    # pipe_flow()
    # solve_aorta2d()
    gm, dPrel, e_uinf, e_ul2, e_pinf, e_pl2, err_ref = aorta_narrowed()

    plot_errors(gm, dPrel, e_uinf, e_ul2, e_pinf, e_pl2, err_ref)
