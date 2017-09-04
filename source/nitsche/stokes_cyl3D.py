''' 3D pipe flow with Navier-Slip (Robin type) boundary conditions
    author: David Nolte
    email: dnolte@dim.uchile.cl

    TODO: check dependence on beta (Nitsche parameter) and alternative
    formulations
'''
# TODO: PCFieldSplitSetBlockSize -> sub blocks? ->
# https://www.mcs.anl.gov/petsc/petsc-current/src/ksp/ksp/examples/tutorials/ex43.c.html,
# l. 1430

from dolfin import *
# petsc4py for petsc fieldsplit block preconditioner
from petsc4py import PETSc
# import numpy as np
# import matplotlib.pyplot as plt
# import scipy.linalg as la
# from sympy.utilities.codegen import ccode
from functions.geom import *
from mpi4py import MPI

import platform
import re
import shutil
import os
import sys

"""
                --petsc.pc_type fieldsplit
                --petsc.pc_fieldsplit_detect_saddle_point
                --petsc.pc_fieldsplit_type schur
                --petsc.pc_fieldsplit_schur_factorization_type diag
                --petsc.pc_fieldsplit_schur_precondition user
                """
# /* mpiexec -n 2 ./ex70 -nx 32 -ny 48 -ksp_type fgmres -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_fact_type lower -fieldsplit_0_ksp_type gmres -fieldsplit_0_pc_type bjacobi -fieldsplit_1_pc_type jacobi -fieldsplit_1_inner_ksp_type preonly -fieldsplit_1_inner_pc_type jacobi -fieldsplit_1_upper_ksp_type preonly -fieldsplit_1_upper_pc_type jacobi */
#  48: /*                                                                             */
#  49: /*   Out-of-the-box SIMPLE-type preconditioning. The major advantage           */
#  50: /*   is that the user neither needs to provide the approximation of            */
#  51: /*   the Schur complement, nor the corresponding preconditioner.               */
args = [sys.argv[0]] + """
            --petsc.ksp_view
            --petsc.ksp_converged_reason
            --petsc.ksp_type fgmres
            --petsc.ksp_rtol 1.0e-8
            --petsc.ksp_monitor

            --petsc.fieldsplit_0_ksp_type preonly
            --petsc.fieldsplit_0_pc_type ml
            --petsc.fieldsplit_0_mg_levels_ksp_type gmres
            --petsc.fieldsplit_0_mg_levels_pc_type bjacobi
            --petsc.fieldsplit_0_mg_levels_ksp_max_it 4

            --petsc.fieldsplit_1_ksp_type preonly
            --petsc.fieldsplit_1_pc_type jacobi
            """.split()
parameters.parse(args)

uname = platform.uname()
pattern = re.compile("^(leftraru\d)|(cn\d\d\d)")
on_cluster = pattern.match(uname[1])

parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "quadrature"
if on_cluster:
    parameters["form_compiler"]["cpp_optimize_flags"] = "-O3 -xHost -ipo"
else:
    # parameters['num_threads'] = 2
    parameters["form_compiler"]["cpp_optimize_flags"] = \
        "-O3 -ffast-math -march=native"


def stokes3D(mesh, beta=1000, solver='lu', elements='TH', gamma=0.0, prec='',
             poiseuille=False, wall='noslip', uwall=0.0, bctype='dirichlet',
             plots=False, inlet='parabola', Rref=None,
             umax=1.):
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
xx          poiseuille: bool, compare with exact poiseuille channel solution
xx          plots:      bool, plot solution
            inlet:      inlet velocity profile, parabola or const
            Rref:       reference radius for inlet parabolic profile & bcs
            umax:       max. velocity for inflow
        output:
            u1, p1      solution
            W.dim()     number of DOFs
    '''
    extmesh = None
    if type(mesh) == str:
        from functions.inout import read_mesh
        mesh, _, bnds = read_mesh(mesh)
        extmesh = True
        # plot(mesh)
        # plot(bnds)

    ndim = mesh.geometry().dim()

    mu = 0.01

    k = Constant(mu)
    beta = Constant(beta)

    # Taylor-Hood Elements
    # if periodic_inlet:
    #     pbc = PeriodicBoundaryINLET(0.2)
    #     V = VectorFunctionSpace(mesh, "CG", 2, constrained_domain=pbc)
    #     Q = FunctionSpace(mesh, "CG", 1, constrained_domain=pbc)
    # else:
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
        W = (P1 + B)*Q

    # boundary numbering
    # 1   inlet
    nin = 1
    # 2   outlet
    nout = 2
    # 3   wall
    ni = 3
    # 4   slip/symmetry
    nis = 4

    if not extmesh:
        coords = mesh.coordinates()
        xmin = coords[:, 0].min()
        xmax = coords[:, 0].max()
        left = Left(xmin)
        right = Right(xmax)
        allbnds = AllBoundaries()

        bnds = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
        bnds.set_all(0)
        allbnds.mark(bnds, ni)
        left.mark(bnds, nin)
        right.mark(bnds, nout)

    ds = Measure('ds', domain=mesh, subdomain_data=bnds)

    zero = Constant((0,)*ndim)

    bcs = []
    # inflow velocity profile
    if inlet == 'parabola':
        # y = a*(((xmax+xmin)/2 - x)**2 -(xmax-xmin)**2/4)
        if Rref or extmesh:
            bmin = -Rref
            bmax = Rref
        else:
            # min/max coords on inlet boundary
            bmin = coords[coords[:, 0] == xmin, 1].min()
            bmax = coords[coords[:, 0] == xmin, 1].max()
        ys = (bmax + bmin) / 2.
        dy = (bmax - bmin) / 2.
        G = -umax/dy**2
        inflow = Expression(('G*(pow(ys - x[1], 2) + pow(ys - x[2], 2) - \
                             pow(dy, 2))', '0.0', '0.0'), ys=ys, dy=dy, G=G)
        # E = interpolate(inflow, V)
        # plot(E)
    elif inlet == 'const':
        inflow = Constant((umax, 0.0, 0.0))

    if elements == 'Mini':
        parameters['krylov_solver']['monitor_convergence'] = False
        inflow = project(inflow, W.sub(0).collapse())

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
        return dot(u, dot(k*grad(v), n))

    def f(v):
        return dot(F, v)*dx

    def t(u, p):
        return dot(k*grad(u), n) - p*n

    # Standard form Stokes equations, Dirichlet RBs
    # XXX ATTENTION! positivity of form not guaranteed -> CHECK!
    a1 = a(u, v) - b(v, p) - b(u, q)
    L1 = f(v)

    # inflow
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

    if extmesh:     # assume symmetric boundary for nis=4 !
        # symmetric BCs (==slip)
        a1 += beta/h*k*dot(u, n)*dot(v, n)*ds(nis) \
            - dot(n, t(u, p))*dot(v, n)*ds(nis)
        #     - dot(u, n)*(k*dot(dot(n, grad(v)), n) + q)*ds(nis) \

    if wall == 'navierslip':
        # normal component of standard boundary integral
        a1 += -(k*dot(dot(n, grad(u)), n) - p)*dot(v, n)*ds(ni)
        # tangential component replaced by Navier-Slip "gamma" BC
        #  (using  n.sigma.t = gamma u.t  and  v = v.n*n + v.t*t)
        a1 += -gamma*k*dot(u - dot(u, n)*n, v - dot(v, n)*n)*ds(ni)
        # Nitsche terms: weakly impose u.n = 0   (tg. comp. taken care of by
        #    Navier-Slip BC) and add skew-symmetric term
        a1 += beta/h*k*dot(u, n)*dot(v, n)*ds(ni) \
            - dot(u, n)*(k*dot(dot(n, grad(v)), n) - q)*ds(ni)

    if wall == 'none':
        # "do nothing" boundary condition
        a1 += -dot(t(u, p), v)*ds(ni)

    if wall == 'noslip':
        noslip = Constant((uwall, 0.0, 0.0))
        if elements == 'Mini':
            parameters['krylov_solver']['monitor_convergence'] = False
            noslip = project(noslip, W.sub(0).collapse())

        if bctype == 'dirichlet':
            bc0 = DirichletBC(W.sub(0), noslip, bnds, ni)
            bcs.append(bc0)
        elif bctype == 'nitsche':
            a1 += - dot(t(u, p), v)*ds(ni)
            a1 += beta/h*k*dot(u, v)*ds(ni)
            L1 += beta/h*k*dot(noslip, v)*ds(ni)
            # consistent positive definite terms
            a1 += - dot(u, t(v, q))*ds(ni)
            L1 += - dot(noslip, t(v, q))*ds(ni)

    if wall == 'slip':
        # u.n = 0, t-(t.n)n = 0
        # normal component is 0 anyways for Dirichlet BC on u_1 !!

        # impose u.n = 0   (=> add "nothing" on rhs)
        a1 += beta/h*k*dot(u, n)*dot(v, n)*ds(ni) \
            - dot(u, n)*(k*dot(dot(n, grad(v)), n) - q)*ds(ni)
        a1 += -dot(n, t(u, p))*dot(v, n)*ds(ni)

    # SUPG
    # alpha = Constant(1./10)
    # a1 += alpha*h**2*dot(-grad(p), grad(q))*dx
    # L1 += -alpha*h**2*dot(F, grad(q))*dx

    # Solve

    w1 = Function(W)
    if solver in ['krylov', 'ksp']:
        # parameters['krylov_solver']['monitor_convergence'] = True
        # parameters["krylov_solver"]["absolute_tolerance"] = 1.0e-8
        # parameters["krylov_solver"]["relative_tolerance"] = 1.0e-6
        if prec == 'schur':
            # PETSc Fieldsplit approach via petsc4py interface
            # Schur complement method
            # create petsc matrices
            # A, b = assemble_system(a1, L1, bcs)
            A = assemble(a1)  # XXX TEST THIS
            b = assemble(L1)
            [bc.apply(A, b) for bc in bcs]
            A = as_backend_type(A).mat()
            b = as_backend_type(b).vec()

            # define Schur complement approximation form
            dstab = Constant(1.0)
            # schur = dstab/k*h**2*dot(grad(p), grad(q))*dx
            schur = dstab/k*p*q*dx
            Sp = assemble(schur)
            Sp = as_backend_type(Sp).mat()

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
            pc.setFieldSplitSchurFactType(PETSc.PC.SchurFactType.DIAG)    # <diag,lower,upper,full>
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
            dstab = Constant(1.)
            # bp1 = a1 + dstab/k*h**2*dot(grad(p), grad(q))*dx
            bp1 = a1 + dstab/k*p*q*dx
            A, b = assemble_system(a1, L1, bcs)
            P, _ = assemble_system(bp1, L1, bcs)
            # [bc.apply(A, P, b) for bc in bcs]

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
            dstab = 10.
            bp1 = a1 + dstab/k*h**2*dot(grad(p), grad(q))*dx
            # bp1 = a1 + dstab/k*p*q*dx
            A, b = assemble_system(a1, L1, bcs)
            P, _ = assemble_system(bp1, L1, bcs)

            solv = KrylovSolver(solver, prec)
            solv.set_operators(A, P)
            solv.solve(w1.vector(), b)
        (u1, p1) = w1.split(deepcopy=True)
    elif solver in ['mumps', 'lu', 'supderlu_dist', 'umfpack', 'petsc']:
        # use direct solver
        # A, b = assemble_system(a1, L1, bcs)
        A = assemble(a1)  # XXX TEST THIS
        b = assemble(L1)
        [bc.apply(A, b) for bc in bcs]

        solve(A, w1.vector(), b, solver)

        # solve(a1 == L1, w1, bcs, solver_parameters={'linear_solver': solver})

        # Split the mixed solution
        (u1, p1) = w1.split(deepcopy=True)

    if elements == 'Mini':
        parameters['krylov_solver']['monitor_convergence'] = False
        E = project(inflow, W.sub(0).collapse())
    else:
        E = interpolate(inflow, W.sub(0).collapse())
    err = norm(u1.vector() - E.vector(), 'l2')
    err_rel = err/norm(E.vector(), 'l2')

    # errnorm = errornorm(inflow, u1)

    unorm = u1.vector().norm("l2")
    pnorm = p1.vector().norm("l2")

    if MPI.COMM_WORLD.rank == 0:
        print("Norm of velocity coefficient vector: %.6g" % unorm)
        print("Norm of pressure coefficient vector: %.6g" % pnorm)
        print("errors @ %d DOFS" % W.dim())
        print("L2 error:\t %f" % err)
        print("L2  rel.:\t %f" % err_rel)
        # print("error norm:\t %f" % errnorm)

    if plots:
        plot(u1, title='velocity, dirichlet')
        plot(p1, title='pressure, dirichlet')

    return u1, p1, W.dim()


def pipe_flow():

    # N = 2**5
    R = 1.1
    Ri = 1.0
    # L = 2

    gamma = 2.*Ri/(Ri**2 - R**2)
    beta = 20.
    umax = 3.

    # mesh = pipe_mesh3D(Ri, L, N)

    mesh_reflvl = 1
    mesh_file = 'pipe4s_r%d.h5' % mesh_reflvl

    if on_cluster:
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
    u1, p1, dof = stokes3D(mesh,
                           wall='navierslip',
                           uwall=uwall(umax, R, Ri),  # noslip only
                           bctype='nitsche',
                           elements='TH',
                           gamma=gamma,            # navier-slip only
                           beta=beta,              # Nitsche only
                           solver='ksp',
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

    if on_cluster:
        # delete mesh from /dev/shm when done
        os.remove(mesh)


if __name__ == '__main__':
    pipe_flow()
