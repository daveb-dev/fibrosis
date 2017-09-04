''' 3D pipe flow with Navier-Slip (Robin type) boundary conditions
    author: David Nolte
    email: dnolte@dim.uchile.cl

    TODO: check dependence on beta (Nitsche parameter) and alternative
    formulations
'''
# TODO: PCFieldSplitSetBlockSize -> sub blocks? ->
# https://www.mcs.anl.gov/petsc/petsc-current/src/ksp/ksp/examples/tutorials/ex43.c.html,
# l. 1430

# TODO: exact algebra, compute directly via numpy/scipy

from dolfin import *
# petsc4py for petsc fieldsplit block preconditioner
from petsc4py import PETSc
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sl
import matplotlib.pyplot as plt
# import scipy.linalg as la
# from sympy.utilities.codegen import ccode
from functions.geom import *
from mpi4py import MPI

from functions.utils import on_cluster

import shutil


parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "quadrature"

if on_cluster():
    parameters["form_compiler"]["cpp_optimize_flags"] = "-O3 -xHost -ipo"
else:
    # parameters['num_threads'] = 2
    parameters["form_compiler"]["cpp_optimize_flags"] = \
        "-O3 -ffast-math -march=native"


def prep_mesh(mesh_file):
    if on_cluster():
        # runbnd_ing on NLHPC cluster
        mfile = '/home/dnolte/fenics/nitsche/meshes/' + mesh_file
        mesh = '/dev/shm/' + mesh_file
        shutil.copy(mfile, '/dev/shm/')
    else:
        # runbnd_ing on local machine
        mesh = 'meshes/' + mesh_file

    return mesh


def uwall(umax, R, y):
    return -umax/R**2*(y**2 - R**2)


def getLS(mesh, beta=10, elements='TH', gamma=0.0, wall='noslip', uwall=0.0,
          bctype='nitsche', solver='mumps', prec='', symmetric=False,
          symmetric_system=False, blockLS=False, A_posdef=False,
          plots=False, inlet='parabola', Rref=None, mu=0.01, umax=1.,):
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
    # plot(mesh)
    ndim = mesh.geometry().dim()

    k = Constant(mu)
    beta = Constant(beta)
    gamma = Constant(gamma)

    if elements == 'TH':
        V = VectorFunctionSpace(mesh, "CG", 2)
        Q = FunctionSpace(mesh, "CG", 1)
        W = V*Q
    elif elements == 'Mini':
        # P1 = VectorElement("CG", mesh.ufl_cell(), 1)
        # B = VectorElement("Bubble", mesh.ufl_cell(), 3)
        # Q = FiniteElement("CG", mesh.ufl_cell(), 1)
        # W = FunctionSpace(mesh, (P1 + B)*Q)
        P1 = VectorFunctionSpace(mesh, "CG", 1)
        B = VectorFunctionSpace(mesh, "Bubble", 4)
        Q = FunctionSpace(mesh, "CG", 1)
        V = P1 + B
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
    if blockLS:
        u = TrialFunction(V)
        v = TestFunction(V)
        p = TrialFunction(Q)
        q = TestFunction(Q)
    else:
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
        return dot(u, dot(k*grad(v), n))

    def f(v):
        return dot(F, v)*dx

    def t(u, p):
        return dot(k*grad(u), n) - p*n

    # Standard form Stokes equations, blockwise
    if not symmetric_system and not A_posdef:   # skew symmetric symstem, B, B^T
        a00 = a(u, v)
        a10 = b(u, q)
        a01 = - b(v, p)
        a11 = Constant(0)*p*q*dx

        L0 = f(v)
        L1 = Constant(0)*q*dx

        # XXX TODO: SIMPLIFY FORMS --> tt = (I-nn), etc
        # inflow
        if bctype == 'dirichlet':
            bcin = DirichletBC(W.sub(0), inflow, bnds, bnd_in)
            bcs.append(bcin)
        else:
            # stress tensor boundary integral
            a00 += - dot(k*grad(u)*n, v)*ds(bnd_in)
            a01 += dot(p*n, v)*ds(bnd_in)
            # Nitsche: u = inflow
            a00 += beta/h*k*dot(u, v)*ds(bnd_in)
            L0 += beta/h*k*dot(inflow, v)*ds(bnd_in)
            # skew symmetric 'balance' terms
            a00 += - dot(u, k*grad(v)*n)*ds(bnd_in)
            a10 += - dot(u, q*n)*ds(bnd_in)
            L0 += - dot(inflow, k*grad(v)*n)*ds(bnd_in)
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
            a00 += - dot(u, n)*dot(k*grad(v)*n, n)*ds(bnd_s)
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
            a00 += - dot(u, n)*dot(k*grad(v)*n, n)*ds(bnd_w)
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
                noslip = project(noslip, W.sub(0).collapse())

            if bctype == 'dirichlet':
                bc0 = DirichletBC(W.sub(0), noslip, bnds, bnd_w)
                bcs.append(bc0)
            elif bctype == 'nitsche':
                # stress tensor boundary integral
                a00 += - dot(k*grad(u)*n, v)*ds(bnd_w)
                a01 += dot(p*n, v)*ds(bnd_w)
                # Nitsche: u = noslip
                a00 += beta/h*k*dot(u, v)*ds(bnd_w)
                L0 += beta/h*k*dot(noslip, v)*ds(bnd_w)
                # skew symmetric 'balance' terms
                a00 += - dot(u, k*grad(v)*n)*ds(bnd_w)
                a10 += - dot(u, q*n)*ds(bnd_w)
                L0 += - dot(noslip, k*grad(v)*n)*ds(bnd_w)
                L1 += - dot(noslip, q*n)*ds(bnd_w)

    elif not A_posdef:   # symmetric linear system B, B^T
        a00 = a(u, v)
        a10 = - b(u, q)
        a01 = - b(v, p)
        a11 = Constant(0)*p*q*dx

        L0 = f(v)
        L1 = Constant(0)*q*dx

        # XXX TODO: SIMPLIFY FORMS --> tt = (I-nn), etc
        # inflow
        if bctype == 'dirichlet':
            bcin = DirichletBC(W.sub(0), inflow, bnds, bnd_in)
            bcs.append(bcin)
        else:
            # stress tensor boundary integral
            a00 += - dot(k*grad(u)*n, v)*ds(bnd_in)
            a01 += dot(p*n, v)*ds(bnd_in)
            # Nitsche: u = inflow
            a00 += beta/h*k*dot(u, v)*ds(bnd_in)
            L0 += beta/h*k*dot(inflow, v)*ds(bnd_in)
            # skew symmetric 'balance' terms
            a00 += - dot(u, k*grad(v)*n)*ds(bnd_in)
            a10 += dot(u, q*n)*ds(bnd_in)
            L0 += - dot(inflow, k*grad(v)*n)*ds(bnd_in)
            L1 += dot(inflow, q*n)*ds(bnd_in)

        if symmetric:     # assume symmetric boundary for bnd_s=4 !
            # symmetric BCs (==slip)
            # normal component of natural boundary integral
            a00 += - dot(k*grad(u)*n, n)*dot(v, n)*ds(bnd_s)
            a01 += p*dot(v, n)*ds(bnd_s)
            # Nitsche: u.n = 0
            a00 += beta/h*k*dot(u, n)*dot(v, n)*ds(bnd_s)
            # RHS -> zero
            # balance terms:
            a00 += - dot(u, n)*dot(k*grad(v)*n, n)*ds(bnd_s)
            a10 += dot(u, n)*q*ds(bnd_s)

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
            a00 += - dot(u, n)*dot(k*grad(v)*n, n)*ds(bnd_w)
            a10 += dot(u, n)*q*ds(bnd_w)

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
                noslip = project(noslip, W.sub(0).collapse())

            if bctype == 'dirichlet':
                bc0 = DirichletBC(W.sub(0), noslip, bnds, bnd_w)
                bcs.append(bc0)
            elif bctype == 'nitsche':
                # stress tensor boundary integral
                a00 += - dot(k*grad(u)*n, v)*ds(bnd_w)
                a01 += dot(p*n, v)*ds(bnd_w)
                # Nitsche: u = noslip
                a00 += beta/h*k*dot(u, v)*ds(bnd_w)
                L0 += beta/h*k*dot(noslip, v)*ds(bnd_w)
                # skew symmetric 'balance' terms
                a00 += - dot(u, k*grad(v)*n)*ds(bnd_w)
                a10 += dot(u, q*n)*ds(bnd_w)
                L0 += - dot(noslip, k*grad(v)*n)*ds(bnd_w)
                L1 += dot(noslip, q*n)*ds(bnd_w)

    if A_posdef:
        a00 = a(u, v)
        a10 = b(u, q)
        a01 = - b(v, p)
        a11 = Constant(0)*p*q*dx

        L0 = f(v)
        L1 = Constant(0)*q*dx

        # XXX TODO: SIMPLIFY FORMS --> tt = (I-nn), etc
        # inflow
        if bctype == 'dirichlet':
            bcin = DirichletBC(W.sub(0), inflow, bnds, bnd_in)
            bcs.append(bcin)
        else:
            # stress tensor boundary integral
            a00 += - dot(k*grad(u)*n, v)*ds(bnd_in)
            a01 += dot(p*n, v)*ds(bnd_in)
            # Nitsche: u = inflow
            a00 += beta/h*k*dot(u, v)*ds(bnd_in)
            L0 += beta/h*k*dot(inflow, v)*ds(bnd_in)
            # 'balance' terms
            a00 += + dot(u, k*grad(v)*n)*ds(bnd_in)
            # a10 += - dot(u, q*n)*ds(bnd_in)
            a10 += dot(u, q*n)*ds(bnd_in)
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
            # a10 += - dot(u, n)*q*ds(bnd_s)
            a10 += dot(u, n)*q*ds(bnd_s)

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
            # a10 += - dot(u, n)*q*ds(bnd_w)
            a10 += dot(u, n)*q*ds(bnd_w)

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
                noslip = project(noslip, W.sub(0).collapse())

            if bctype == 'dirichlet':
                bc0 = DirichletBC(W.sub(0), noslip, bnds, bnd_w)
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
                # a10 += - dot(u, q*n)*ds(bnd_w)
                a10 += dot(u, q*n)*ds(bnd_w)
                L0 += + dot(noslip, k*grad(v)*n)*ds(bnd_w)
                L1 += -dot(noslip, q*n)*ds(bnd_w)

    A00 = assemble(a00)
    B = assemble(a10)
    BT = assemble(a01)
    b0 = assemble(L0)
    b1 = assemble(L1)

    if not blockLS:
        a1 = a00 + a10 + a01 + a11
        L = L0 + L1
        A = assemble(a1)  # XXX TEST THIS vs. assemble_system
        b = assemble(L)
        [bc.apply(A, b) for bc in bcs]
        [bc.apply(A00, B, BT, b0, b1) for bc in bcs]

        P = None
        if solver in ['krylov', 'ksp']:
            if prec == 'jacobi':
                dstab = Constant(1.)
                # bp1 = a1 + dstab/k*h**2*dot(grad(p), grad(q))*dx
                bp1 = a00 + dstab/k*p*q*dx
                A, b = assemble_system(a1, L1, bcs)
                P, _ = assemble_system(bp1, L1, bcs)
                # [bc.apply(A, P, b) for bc in bcs]
            elif prec == 'schur':
                dstab = Constant(1.0)
                # schur = dstab/k*h**2*dot(grad(p), grad(q))*dx
                schur = dstab/k*p*q*dx
                P = assemble(schur)
            elif prec == 'direct':
                # XXX ACCOUNT FOR CONSTANT PRESSURE NULLSPACE!?
                # cf. Elman (2005) book pp. 83
                dstab = 10.
                bp1 = a00 + dstab/k*h**2*dot(grad(p), grad(q))*dx
                # bp1 = a1 + dstab/k*p*q*dx
                A, b = assemble_system(a1, L1, bcs)
                P, _ = assemble_system(bp1, L1, bcs)
    else:
        A = np.hstack((A00.array(), BT.array()))
        n2 = B.array().shape[0]
        A1 = np.hstack((B.array(), np.zeros((n2, n2))))
        A = np.vstack((A, A1))
        P = None
        print(b0.array().shape)
        b = np.vstack((b0.array().reshape((len(b0.array()), 1)),
                       b1.array().reshape((len(b1.array()), 1))))

    return A, P, b, W, A00, B, BT, b0, b1

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


def solveLS_numpy(A, v, W):
    A = sp.csr_matrix(A)

    # invert A
    Ainv = sl.inv(A)

    # compute Schur complement


if __name__ == '__main__':
    umax = 3.
    R = 1.0
    Ri = 1.0
    mu = 0.01
    gamma = -10.0*mu
    beta = 0.0

    mesh_reflvl = 0
    # mesh_file = 'coarc2d_coarse_r%d.h5' % mesh_reflvl
    mesh_file = 'unit2d_r%d.h5' % mesh_reflvl
    mesh = prep_mesh(mesh_file)

    A, P, b, W, A00, B, BT, b0, b1 = getLS(
        mesh,
        wall='navierslip',
        uwall=uwall(umax, R, Ri),  # noslip only
        bctype='nitsche',
        elements='TH',
        symmetric=True,
        gamma=gamma,            # navier-slip only
        beta=beta,              # Nitsche only
        solver='ksp',
        prec='schur',
        plots=True,
        inlet='parabola',  # parabola, const
        Rref=R,
        mu=mu,
        umax=umax,
        blockLS=True,
        A_posdef=True,   # not positive definite, but Re(eig)>0 ! (??)
        symmetric_system=False,
    )
    # check if pos def -> np.linalg.cholesky(A00.array())

    plt.ion()
    plt.figure()
    plt.spy(A)
    plt.title('K')
    plt.figure()
    plt.spy(A00.array())
    plt.title('A, A00')
    plt.figure()
    plt.spy(BT.array())
    plt.title('BT')
    plt.figure()
    plt.spy(B.array())
    plt.title('B')
